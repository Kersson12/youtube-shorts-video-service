#!/usr/bin/env python3
"""
Video Composition Service
  POST /compose        → genera video via MiniMax Hailuo (fal.ai) + subtitulos FFmpeg
  POST /send-telegram  → envia video a Telegram con botones aprobacion
  POST /upload-youtube → sube video a YouTube
  GET  /health
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import base64, os, tempfile, requests, subprocess, traceback, logging, uuid, json, math, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw
from tts_engine import generate_tts_local

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

VIDEOS_DIR = '/tmp/videos'
os.makedirs(VIDEOS_DIR, exist_ok=True)


WIDTH, HEIGHT = 1080, 1920
FONT_REG = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

TG_TOKEN         = os.environ['TG_TOKEN']
CHAT_ID          = os.environ['CHAT_ID']
YT_CLIENT_ID     = os.environ['YT_CLIENT_ID']
YT_CLIENT_SECRET = os.environ['YT_CLIENT_SECRET']
YT_REFRESH_TOKEN = os.environ['YT_REFRESH_TOKEN']
FAL_API_KEY      = os.environ.get('FAL_API_KEY', '')

# Channel 2 (psicología) — optional, falls back to channel 1 if not set
TG_CHAT_ID_2       = os.environ.get('TG_CHAT_ID_2', '')
YT_REFRESH_TOKEN_2 = os.environ.get('YT_REFRESH_TOKEN_2', '')

def _yt_credentials(channel: str = '1') -> str:
    """Return the refresh token for the given channel slot."""
    if channel == '2' and YT_REFRESH_TOKEN_2:
        return YT_REFRESH_TOKEN_2
    return YT_REFRESH_TOKEN

def _tg_chat_id(channel: str = '1') -> str:
    if channel == '2' and TG_CHAT_ID_2:
        return TG_CHAT_ID_2
    return CHAT_ID

FAL_HEADERS = lambda: {'Authorization': f'Key {FAL_API_KEY}', 'Content-Type': 'application/json'}


def _pexels_search_one(query: str, api_key: str, used_ids: set) -> dict | None:
    """Search Pexels for one unique video. Randomizes page to avoid always getting same clips."""
    for orientation in ('portrait', None):
        page = random.randint(1, 4)
        params = {'query': query, 'per_page': 10, 'size': 'small', 'page': page}
        if orientation:
            params['orientation'] = orientation
        try:
            resp = requests.get(
                'https://api.pexels.com/videos/search',
                params=params, headers={'Authorization': api_key}, timeout=10
            )
            if resp.status_code != 200:
                continue
            videos = resp.json().get('videos', [])
            random.shuffle(videos)  # shuffle so we don't always pick the first result
            for v in videos:
                vid_id = v.get('id')
                if vid_id in used_ids:
                    continue
                files = [f for f in v.get('video_files', [])
                         if f.get('link') and f.get('width', 0) >= 360]
                if files:
                    files.sort(key=lambda f: f.get('width', 9999))
                    used_ids.add(vid_id)
                    return files[0]
        except Exception:
            pass
    return None


def download_pexels_clips(keywords: list, tmp: str, api_key: str) -> list:
    """Download 1 clip per keyword from Pexels, with broad fallbacks to reach 5 clips."""
    # Build search queries: specific keyword, then broader fallbacks
    finance_fallbacks = [
        'money colombia', 'urban street colombia', 'finance city',
        'shopping market', 'bank building', 'city traffic colombia'
    ]
    queries = list(keywords[:5]) + finance_fallbacks

    paths = []
    used_ids: set = set()
    for i, kw in enumerate(queries):
        if len(paths) >= 5:
            break
        file_info = _pexels_search_one(kw, api_key, used_ids)
        if not file_info:
            logger.warning(f'Pexels no results for "{kw}"')
            continue
        url = file_info['link']
        try:
            dl = requests.get(url, timeout=30, stream=True)
            if dl.status_code != 200:
                continue
            p = os.path.join(tmp, f'px_{len(paths)}.mp4')
            with open(p, 'wb') as f:
                for chunk in dl.iter_content(65536):
                    f.write(chunk)
            paths.append(p)
            logger.info(f'Pexels clip {len(paths)}/5 "{kw}": ok')
        except Exception as e:
            logger.warning(f'Pexels download "{kw}": {e}')

    logger.info(f'Pexels total: {len(paths)} clips')
    return paths


# ── Models ────────────────────────────────────────────────────────────────────
class ComposeRequest(BaseModel):
    guion: dict
    audio_base64: str
    alignment: Optional[dict] = None
    word_timestamps: Optional[list] = None   # preferred: [{word, start, end}]
    image_url: Optional[str] = ''

class ComposeResponse(BaseModel):
    file_id: str
    duration_seconds: float
    title: str

class TelegramRequest(BaseModel):
    file_id: str
    resume_url: str
    titulo: str
    nicho: str
    duration: float
    numero_estrella: Optional[str] = ''
    unidad_numero: Optional[str] = ''
    channel: Optional[str] = '1'
    canal_nombre: Optional[str] = ''   # e.g. "🧠 Psicología" or "💰 Finanzas"

class YouTubeRequest(BaseModel):
    file_id: str
    titulo: str
    descripcion: Optional[str] = ''
    hashtags: Optional[List[str]] = []
    channel: Optional[str] = '1'

class YouTubeResponse(BaseModel):
    video_id: str
    video_url: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_audio_duration(path: str) -> float:
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', path],
        capture_output=True, text=True)
    return float(out.stdout.strip())


def build_word_timestamps(alignment: dict) -> list:
    chars  = alignment.get('characters', [])
    starts = alignment.get('character_start_times_seconds', [])
    ends   = alignment.get('character_end_times_seconds', [])
    if not chars:
        return []
    words, cur, ws = [], '', 0.0
    for i, (ch, st, en) in enumerate(zip(chars, starts, ends)):
        last = i == len(chars) - 1
        if ch == ' ' or last:
            if last and ch != ' ':
                cur += ch
                en = ends[i]
            if cur.strip():
                words.append((cur.strip(), ws, en))
            cur, ws = '', en
        else:
            if not cur:
                ws = st
            cur += ch
    return words


def build_ass(words: list, tmp: str) -> str:
    def ts(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f'{h}:{m:02d}:{s:05.2f}'

    header = (
        '[Script Info]\nScriptType: v4.00+\n'
        f'PlayResX: {WIDTH}\nPlayResY: {HEIGHT}\n\n'
        '[V4+ Styles]\n'
        'Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,'
        'OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,'
        'ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,'
        'Alignment,MarginL,MarginR,MarginV,Encoding\n'
        # Fontsize 90, bold, outline 4, shadow 6, pushed up from bottom (MarginV 420)
        'Style: Default,DejaVu Sans Bold,90,&H00FFFFFF,&H000000FF,'
        '&H00000000,&HAA000000,-1,0,0,0,100,100,2,0,1,4,6,'
        '2,60,60,560,1\n\n'
        '[Events]\nFormat: Layer,Start,End,Style,Name,'
        'MarginL,MarginR,MarginV,Effect,Text\n'
    )
    CHUNK = 3  # 3 palabras por linea — mas legible con fuente grande
    events = []
    for i, (word, start, end) in enumerate(words):
        chunk_start = (i // CHUNK) * CHUNK
        chunk = words[chunk_start: chunk_start + CHUNK]
        pos = i - chunk_start
        next_start = words[i + 1][1] if i + 1 < len(words) else end + 0.1
        parts = []
        for j, (w, _, _) in enumerate(chunk):
            if j < pos:
                # ya dicha: blanco semitransparente
                parts.append('{\\c&H99FFFFFF&}' + w + '{\\c&H00FFFFFF&}')
            elif j == pos:
                # activa: AMARILLO brillante — máximo contraste
                parts.append('{\\c&H0000FFFF&\\3c&H00000000&}' + w + '{\\c&H00FFFFFF&\\3c&H00000000&}')
            else:
                # próxima: gris claro
                parts.append('{\\c&H55FFFFFF&}' + w + '{\\c&H00FFFFFF&}')
        events.append(
            f'Dialogue: 0,{ts(start)},{ts(next_start)},'
            f'Default,,0,0,0,,' + ' '.join(parts)
        )
    path = os.path.join(tmp, 'subs.ass')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header + '\n'.join(events))
    return path


def generate_clip_minimax(prompt: str, idx: int) -> tuple:
    """Generate one 6s clip via MiniMax Hailuo on fal.ai (~$0.035/clip).
    Returns (idx, url)."""
    if not FAL_API_KEY:
        raise Exception('FAL_API_KEY not set')

    logger.info(f'fal.ai MiniMax clip {idx}: {prompt[:70]}...')
    resp = requests.post(
        'https://fal.run/fal-ai/minimax/video-01',
        headers=FAL_HEADERS(),
        json={'prompt': prompt, 'aspect_ratio': '9:16'},
        timeout=300
    )
    if resp.status_code != 200:
        raise Exception(f'fal.ai MiniMax {idx} HTTP {resp.status_code}: {resp.text[:300]}')

    data = resp.json()
    video = data.get('video', {})
    url = video.get('url', '') if isinstance(video, dict) else ''
    if not url or not url.startswith('http'):
        raise Exception(f'MiniMax {idx} no URL. Response: {json.dumps(data)[:300]}')

    logger.info(f'fal.ai clip {idx} ready: {url}')
    return idx, url


def _esc(s: str) -> str:
    return (s.replace('\\', '\\\\')
             .replace("'", '\u2019')
             .replace(':', '\\:')
             .replace('%', '%%'))


def normalize_clip(src: str, dst: str) -> bool:
    """Re-encode clip to standard 30fps H264 1080x1920 to prevent concat freezing.
    Different Pexels clips have different codecs/fps/colorspace — must normalize first."""
    r = subprocess.run([
        'ffmpeg', '-y', '-threads', '2', '-i', src,
        '-vf', (f'fps=30,scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,'
                f'crop={WIDTH}:{HEIGHT},setsar=1'),
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '32',
        '-r', '30', '-an', dst
    ], capture_output=True, text=True)
    return r.returncode == 0


def compose_single_pass(clip_urls: list, audio_path: str, ass_path: str,
                        fuente: str, duration: float, tmp: str) -> str:
    """Download + normalize clips, then ONE FFmpeg pass for overlay+subs+audio."""
    raw_paths = []
    for i, url in enumerate(clip_urls):
        if url.startswith('http'):
            path = os.path.join(tmp, f'raw_{i}.mp4')
            dl = requests.get(url, timeout=60, stream=True)
            with open(path, 'wb') as f:
                for chunk in dl.iter_content(65536):
                    f.write(chunk)
            logger.info(f'Downloaded clip {i}: {path}')
        else:
            path = url
            logger.info(f'Local clip {i}: {path}')
        raw_paths.append(path)

    # Normalize every clip to same codec/fps/resolution — eliminates freezing
    norm_paths = []
    for i, path in enumerate(raw_paths):
        norm = os.path.join(tmp, f'norm_{i}.mp4')
        if normalize_clip(path, norm):
            norm_paths.append(norm)
            logger.info(f'Normalized clip {i}')
        else:
            norm_paths.append(path)  # fallback to original
            logger.warning(f'Normalize failed for clip {i}, using original')

    # Loop normalized clips to cover full audio duration
    clip_dur = 5  # seconds per clip after normalization
    clips_needed = int(math.ceil(duration / clip_dur)) + 1
    looped = (norm_paths * (clips_needed // len(norm_paths) + 1))[:clips_needed]

    concat_txt = os.path.join(tmp, 'clips_concat.txt')
    with open(concat_txt, 'w') as f:
        for p in looped:
            f.write(f"file '{p}'\n")

    # Final pass: overlay + subtitles + audio
    # Boost saturation/contrast so clips look vivid, light overlay just for text legibility
    vf_parts = [
        'eq=saturation=1.6:contrast=1.1:brightness=0.04',
        'drawbox=x=0:y=0:w=iw:h=ih:color=black@0.20:t=fill'
    ]
    if fuente:
        vf_parts.append(
            f"drawtext=fontfile='{FONT_REG}':text='{_esc(fuente)}'"
            f":x=40:y=h-70:fontsize=26:fontcolor=white@0.65"
        )
    vf_parts.append(f'ass={ass_path}')

    out = os.path.join(tmp, 'output.mp4')
    r = subprocess.run([
        'ffmpeg', '-y', '-threads', '2',
        '-f', 'concat', '-safe', '0', '-i', concat_txt,
        '-i', audio_path,
        '-map', '0:v:0',   # explicit: video from concat clips only
        '-map', '1:a:0',   # explicit: audio from TTS file only
        '-vf', ','.join(vf_parts),
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
        '-c:a', 'aac', '-b:a', '128k',
        '-pix_fmt', 'yuv420p',
        '-shortest', out
    ], capture_output=True, text=True)
    if r.returncode != 0:
        raise Exception(f'FFmpeg compose failed:\n{r.stderr[-600:]}')
    return out


def fallback_pillow_bg(tmp: str) -> str:
    """Dark gradient fallback background."""
    img = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        t = y / HEIGHT
        draw.line([(0, y), (WIDTH, y)], fill=(int(5+t*8), int(12+t*20), int(8+t*12)))
    path = os.path.join(tmp, 'bg_fallback.png')
    img.save(path, 'PNG')
    return path


# ── /compose ──────────────────────────────────────────────────────────────────
@app.post('/compose', response_model=ComposeResponse)
async def compose_video(req: ComposeRequest):
    try:
        with tempfile.TemporaryDirectory() as tmp:
            logger.info('Composing video...')

            # Save audio
            audio_path = os.path.join(tmp, 'audio.mp3')
            with open(audio_path, 'wb') as f:
                f.write(base64.b64decode(req.audio_base64))
            duration = get_audio_duration(audio_path)

            # Build subtitles — prefer word_timestamps (direct), fall back to alignment
            if req.word_timestamps:
                words = [(w['word'], w['start'], w['end']) for w in req.word_timestamps]
            else:
                words = build_word_timestamps(req.alignment or {})
            ass_path = build_ass(words, tmp)
            fuente   = str(req.guion.get('fuente', ''))[:55].strip()

            # Build visual prompts from guion keywords
            keywords = req.guion.get('visual_keywords', [])
            nicho    = req.guion.get('nicho', 'finanzas Colombia')
            kw_str   = ', '.join(keywords[:4]) if keywords else nicho

            # PRIMARY: Pexels (free) — 1 clip per keyword, real stock footage
            # FALLBACK: fal.ai MiniMax — only if Pexels finds nothing
            clip_urls = []
            pexels_key = os.environ.get('PEXELS_API_KEY', '')
            if pexels_key and keywords:
                logger.info('Fetching Pexels clips (free)...')
                clip_urls = download_pexels_clips(keywords, tmp, pexels_key)
                logger.info(f'Pexels: {len(clip_urls)} clips')

            if not clip_urls:
                logger.info('Pexels unavailable — using fal.ai MiniMax fallback...')
                base_style = "vertical portrait 9:16 video, dark moody finance, no text, no faces"
                kws = (keywords + [nicho] * 3)[:3]
                prompts = [
                    f"{kws[0]}, {base_style}, dramatic shot, Colombia urban",
                    f"{kws[1]}, {base_style}, slow push-in, bokeh",
                    f"{kws[2]}, {base_style}, close-up, cinematic grade",
                ]
                for i, prompt in enumerate(prompts):
                    try:
                        _, url = generate_clip_minimax(prompt, i)
                        clip_urls.append(url)
                    except Exception as e:
                        logger.error(f'fal.ai clip {i} failed: {e}')

            if not clip_urls:
                raise Exception('Pexels y fal.ai fallaron. Verifica PEXELS_API_KEY y saldo fal.ai')

            # Single FFmpeg pass: concat+scale+overlay+subs+audio
            tmp_out = compose_single_pass(clip_urls, audio_path, ass_path, fuente, duration, tmp)

            file_id    = str(uuid.uuid4())
            final_path = os.path.join(VIDEOS_DIR, f'{file_id}.mp4')
            os.rename(tmp_out, final_path)
            logger.info(f'Video saved: {final_path}')

            return ComposeResponse(
                file_id=file_id,
                duration_seconds=round(duration, 2),
                title=req.guion.get('titulo', 'short')
            )
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f'COMPOSE ERROR:\n{tb}')
        raise HTTPException(status_code=500, detail=f'{e}\n\n{tb}')


# ── /send-telegram ───────────────────────────────────────────────────────────��
@app.post('/send-telegram')
async def send_telegram(req: TelegramRequest):
    try:
        video_path = os.path.join(VIDEOS_DIR, f'{req.file_id}.mp4')
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f'Video not found: {req.file_id}')

        chat_id = _tg_chat_id(req.channel or '1')
        canal_label = req.canal_nombre or ('🧠 Psicología' if (req.channel or '1') == '2' else '💰 Finanzas')
        caption = (
            f'{canal_label} — Video listo para publicar\n\n'
            f'Titulo: {req.titulo}\n'
            f'Nicho: {req.nicho}\n'
            f'Duracion: {req.duration:.1f}s'
        )
        with open(video_path, 'rb') as vf:
            r = requests.post(
                f'https://api.telegram.org/bot{TG_TOKEN}/sendVideo',
                data={'chat_id': chat_id, 'caption': caption},
                files={'video': ('short.mp4', vf, 'video/mp4')},
                timeout=120
            )
        if r.status_code != 200:
            raise Exception(f'Telegram sendVideo failed: {r.text[:300]}')

        r2 = requests.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={
                'chat_id': chat_id,
                'text': 'Publicar este video en YouTube?',
                'reply_markup': {
                    'inline_keyboard': [[
                        {'text': 'SI - Publicar', 'url': req.resume_url + '?approve=1'},
                        {'text': 'NO - Rechazar', 'url': req.resume_url + '?approve=0'}
                    ]]
                }
            },
            timeout=15
        )
        logger.info(f'Telegram sent: {r2.status_code}')
        return {'sent': True}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f'TELEGRAM ERROR:\n{tb}')
        raise HTTPException(status_code=500, detail=str(e))


# ── /upload-youtube ───────────────────────────────────────────────────────────
@app.post('/upload-youtube', response_model=YouTubeResponse)
async def upload_youtube(req: YouTubeRequest):
    try:
        video_path = os.path.join(VIDEOS_DIR, f'{req.file_id}.mp4')
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f'Video not found: {req.file_id}')

        token_r = requests.post('https://oauth2.googleapis.com/token', data={
            'client_id': YT_CLIENT_ID,
            'client_secret': YT_CLIENT_SECRET,
            'refresh_token': _yt_credentials(req.channel or '1'),
            'grant_type': 'refresh_token'
        }, timeout=15)
        access_token = token_r.json().get('access_token')
        if not access_token:
            raise Exception(f'No access token: {token_r.text[:200]}')

        file_size = os.path.getsize(video_path)
        titulo    = (req.titulo or 'Short financiero')[:97].strip()
        # YouTube title limit is 100 chars; strip < > which YouTube rejects
        titulo    = titulo.replace('<', '').replace('>', '')
        desc      = (req.descripcion or '') + '\n\n' + ' '.join(req.hashtags or [])
        tags      = [h.replace('#', '') for h in (req.hashtags or [])]

        init_r = requests.post(
            'https://www.googleapis.com/upload/youtube/v3/videos'
            '?uploadType=resumable&part=snippet,status',
            headers={
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'X-Upload-Content-Type': 'video/mp4',
                'X-Upload-Content-Length': str(file_size)
            },
            json={
                'snippet': {'title': titulo, 'description': desc, 'tags': tags,
                            'categoryId': '22', 'defaultLanguage': 'es'},
                'status': {'privacyStatus': 'public', 'selfDeclaredMadeForKids': False}
            },
            timeout=30
        )
        upload_url = init_r.headers.get('location')
        if not upload_url:
            raise Exception(f'No upload URL: {init_r.text[:200]}')

        with open(video_path, 'rb') as vf:
            upload_r = requests.put(
                upload_url,
                headers={'Content-Type': 'video/mp4', 'Content-Length': str(file_size)},
                data=vf, timeout=300
            )
        video_id = upload_r.json().get('id')
        if not video_id:
            raise Exception(f'No video ID: {upload_r.text[:300]}')

        os.remove(video_path)
        logger.info(f'YouTube upload done: {video_id}')
        return YouTubeResponse(
            video_id=video_id,
            video_url=f'https://youtube.com/shorts/{video_id}'
        )
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f'YOUTUBE ERROR:\n{tb}')
        raise HTTPException(status_code=500, detail=str(e))


# ── /tts — Google Cloud TTS Studio (v1beta1 con timestamps reales) ───────────
GOOGLE_TTS_API_KEY = os.environ.get('GOOGLE_TTS_API_KEY', '')

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = 'es-US-Studio-B'
    rate: Optional[str] = '-5%'   # slightly slower than normal for clarity

class TTSResponse(BaseModel):
    audio_base64: str
    word_timestamps: list

def _rate_to_float(rate_str: str) -> float:
    rate_str = rate_str.strip()
    if rate_str.endswith('%'):
        pct = float(rate_str[:-1])
        return round(1.0 + pct / 100.0, 2)
    try:
        return float(rate_str)
    except Exception:
        return 1.05

def _google_tts_call(api_key: str, payload: dict, beta: bool = False) -> dict:
    version = 'v1beta1' if beta else 'v1'
    url = f'https://texttospeech.googleapis.com/{version}/text:synthesize?key={api_key}'
    resp = requests.post(url, json=payload, timeout=30)
    if resp.status_code != 200:
        raise Exception(f'Google TTS error {resp.status_code}: {resp.text[:400]}')
    return resp.json()

@app.post('/tts', response_model=TTSResponse)
async def text_to_speech(req: TTSRequest):
    try:
        logger.info(f"Generando TTS local con Kokoro para: {req.text[:50]}...")
        audio_b64, word_timestamps = generate_tts_local(
            text=req.text,
            voice=req.voice if req.voice else "af_heart", # El motor mapeará a español si detecta el idioma
            rate=req.rate or "-5%"
        )
        return TTSResponse(audio_base64=audio_b64, word_timestamps=word_timestamps)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f'TTS ERROR:\n{tb}')
        raise HTTPException(status_code=500, detail=str(e))


# ── /upload-instagram ────────────────────────────────────────────────────────
IG_USER_ID    = os.environ.get('INSTAGRAM_USER_ID', '')
IG_TOKEN      = os.environ.get('INSTAGRAM_ACCESS_TOKEN', '')
TIKTOK_TOKEN  = os.environ.get('TIKTOK_ACCESS_TOKEN', '')
TIKTOK_OPENID = os.environ.get('TIKTOK_OPEN_ID', '')

class SocialRequest(BaseModel):
    file_id: str
    titulo: str
    hashtags: Optional[List[str]] = []

@app.post('/upload-instagram')
async def upload_instagram(req: SocialRequest):
    if not IG_USER_ID or not IG_TOKEN:
        raise HTTPException(status_code=501, detail='INSTAGRAM_USER_ID / INSTAGRAM_ACCESS_TOKEN no configurados')
    try:
        video_path = os.path.join(VIDEOS_DIR, f'{req.file_id}.mp4')
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f'Video no encontrado: {req.file_id}')

        file_size = os.path.getsize(video_path)
        caption   = req.titulo + '\n\n' + ' '.join(req.hashtags or [])
        graph_url = f'https://graph.instagram.com/v21.0/{IG_USER_ID}'

        # 1. Crear sesion de upload resumable
        init_r = requests.post(f'{graph_url}/media', params={
            'media_type': 'REELS',
            'upload_type': 'resumable',
            'caption': caption,
            'share_to_feed': 'true',
            'access_token': IG_TOKEN,
        }, timeout=30)
        init_data = init_r.json()
        upload_url = init_data.get('uri') or init_data.get('upload_url')
        creation_id = init_data.get('id')
        if not upload_url or not creation_id:
            raise Exception(f'Instagram init failed: {init_r.text[:300]}')

        # 2. Subir archivo
        with open(video_path, 'rb') as vf:
            up_r = requests.post(upload_url, headers={
                'Authorization': f'OAuth {IG_TOKEN}',
                'offset': '0',
                'file_size': str(file_size),
            }, data=vf, timeout=300)
        if up_r.status_code not in (200, 201):
            raise Exception(f'Instagram upload failed: {up_r.text[:300]}')

        # 3. Publicar
        pub_r = requests.post(f'{graph_url}/media_publish', params={
            'creation_id': creation_id,
            'access_token': IG_TOKEN,
        }, timeout=30)
        media_id = pub_r.json().get('id')
        if not media_id:
            raise Exception(f'Instagram publish failed: {pub_r.text[:300]}')

        logger.info(f'Instagram Reel published: {media_id}')
        return {'instagram_id': media_id, 'url': f'https://www.instagram.com/reel/{media_id}/'}
    except Exception as e:
        logger.error(f'INSTAGRAM ERROR: {e}')
        raise HTTPException(status_code=500, detail=str(e))


# ── /upload-tiktok ────────────────────────────────────────────────────────────
@app.post('/upload-tiktok')
async def upload_tiktok(req: SocialRequest):
    if not TIKTOK_TOKEN or not TIKTOK_OPENID:
        raise HTTPException(status_code=501, detail='TIKTOK_ACCESS_TOKEN / TIKTOK_OPEN_ID no configurados')
    try:
        video_path = os.path.join(VIDEOS_DIR, f'{req.file_id}.mp4')
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f'Video no encontrado: {req.file_id}')

        file_size = os.path.getsize(video_path)
        title     = req.titulo[:150]  # TikTok max 150 chars

        # 1. Inicializar upload
        init_r = requests.post(
            'https://open.tiktokapis.com/v2/post/publish/video/init/',
            headers={'Authorization': f'Bearer {TIKTOK_TOKEN}', 'Content-Type': 'application/json'},
            json={
                'post_info': {
                    'title': title,
                    'privacy_level': 'PUBLIC_TO_EVERYONE',
                    'disable_duet': False,
                    'disable_comment': False,
                    'disable_stitch': False,
                },
                'source_info': {
                    'source': 'FILE_UPLOAD',
                    'video_size': file_size,
                    'chunk_size': file_size,
                    'total_chunk_count': 1,
                },
            }, timeout=30
        )
        init_data = init_r.json().get('data', {})
        upload_url = init_data.get('upload_url')
        publish_id = init_data.get('publish_id')
        if not upload_url:
            raise Exception(f'TikTok init failed: {init_r.text[:300]}')

        # 2. Subir video en un chunk
        with open(video_path, 'rb') as vf:
            up_r = requests.put(upload_url, headers={
                'Content-Type': 'video/mp4',
                'Content-Range': f'bytes 0-{file_size-1}/{file_size}',
                'Content-Length': str(file_size),
            }, data=vf, timeout=300)
        if up_r.status_code not in (200, 201, 206):
            raise Exception(f'TikTok upload failed: {up_r.text[:300]}')

        logger.info(f'TikTok upload done: publish_id={publish_id}')
        # Eliminar video despues del ultimo upload
        try:
            os.remove(video_path)
        except Exception:
            pass
        return {'publish_id': publish_id}
    except Exception as e:
        logger.error(f'TIKTOK ERROR: {e}')
        raise HTTPException(status_code=500, detail=str(e))


# ── /reject ───────────────────────────────────────────────────────────────────
@app.post('/reject')
async def reject_video(body: dict):
    file_id = body.get('file_id', '')
    video_path = os.path.join(VIDEOS_DIR, f'{file_id}.mp4')
    if os.path.exists(video_path):
        os.remove(video_path)
    requests.post(
        f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
        json={'chat_id': CHAT_ID, 'text': 'Video rechazado — no se publicó en YouTube.'},
        timeout=10
    )
    return {'rejected': True}


# ── Memory (Safe Persistence) ──────────────────────────────────────────────────
MEMORY_DIR = os.path.expanduser('~/shorts_data')
os.makedirs(MEMORY_DIR, exist_ok=True)
MEMORY_FILE = os.path.join(MEMORY_DIR, 'memory.json')

def _load_memory():
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'uploaded': [], 'rejected': []}

def _save_memory(mem):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)

@app.get('/memory')
def get_memory():
    return _load_memory()

@app.post('/memory/uploaded')
async def record_uploaded(body: dict):
    mem = _load_memory()
    mem['uploaded'].append({k: body.get(k, '') for k in ('titulo', 'nicho', 'video_url', 'date')})
    mem['uploaded'] = mem['uploaded'][-30:]
    _save_memory(mem)
    return {'ok': True}

@app.post('/memory/rejected')
async def record_rejected(body: dict):
    mem = _load_memory()
    titulo = body.get('titulo', '')
    for item in mem['rejected']:
        if item.get('titulo') == titulo:
            item['count'] = item.get('count', 1) + 1
            _save_memory(mem)
            return {'ok': True}
    mem['rejected'].append({k: body.get(k, '') for k in ('titulo', 'nicho', 'date')})
    mem['rejected'] = mem['rejected'][-20:]
    _save_memory(mem)
    return {'ok': True}


@app.get('/health')
def health():
    return {'status': 'ok'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
