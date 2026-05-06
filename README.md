# bubble-room-realtime-social-ai
FastAPI realtime social room prototype with WebSocket chat, JWT auth, Redis pub/sub, AI host, matchmaking, WebRTC signaling, and MediaPipe avatar sync.
#!/usr/bin/env python3
"""
Bubble Room — Realtime Social AI Room Prototype

Features:
- FastAPI WebSocket server
- JWT authentication
- Redis Pub/Sub global chat
- AI host assistant with Gemini optional integration
- Interest-based 1v1 matchmaking
- WebRTC signaling channel
- MediaPipe face blendshape avatar sync
- Rate limiting with Redis
- XP-based friendly challenge system

Disclaimer:
This is a portfolio/research prototype. It does not include real-money betting,
payment processing, or production-grade moderation.
"""

import asyncio
import base64
import json
import os
import time
import uuid
from typing import Dict, List, Optional, Any

import cv2
import jwt
import mediapipe as mp
import numpy as np
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel
from sqlalchemy import Column, Float, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception:
    python = None
    vision = None


# ==============================================================================
# CONFIG
# ==============================================================================

SECRET_KEY = os.getenv("BUBBLE_SECRET_KEY", "dev_secret_change_me")
ALGORITHM = "HS256"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://bubble_user:bubble_password@localhost:5432/bubbleroom",
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

FACE_LANDMARKER_PATH = os.getenv("FACE_LANDMARKER_PATH", "face_landmarker.task")

RATE_LIMIT_PER_SECOND = int(os.getenv("RATE_LIMIT_PER_SECOND", "5"))

app = FastAPI(
    title="Bubble Room Realtime Social AI",
    version="1.0.0",
)


# ==============================================================================
# DATABASE
# ==============================================================================

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


class UserDB(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    rank = Column(String, default="Bronze")
    xp = Column(Integer, default=0)
    virtual_credits = Column(Float, default=0.0)


# ==============================================================================
# REDIS
# ==============================================================================

redis_client = redis.from_url(REDIS_URL, decode_responses=True)


async def check_rate_limit(user_id: str, limit: int = RATE_LIMIT_PER_SECOND, window: int = 1) -> bool:
    key = f"rate_limit:{user_id}"
    current = await redis_client.incr(key)

    if current == 1:
        await redis_client.expire(key, window)

    return current <= limit


# ==============================================================================
# AUTH
# ==============================================================================

class LoginRequest(BaseModel):
    user_id: str


def create_token(user_id: str, expires_in_sec: int = 36000) -> str:
    payload = {
        "sub": user_id,
        "exp": int(time.time()) + expires_in_sec,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


@app.post("/login")
async def login(req: LoginRequest):
    token = create_token(req.user_id)
    return {
        "ok": True,
        "token": token,
        "message": "Login successful.",
    }


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "bubble-room",
    }


# ==============================================================================
# AI HOST
# ==============================================================================

class MaydaAIHost:
    def __init__(self):
        self.model = None

        if genai and GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
            except Exception:
                self.model = None

    def get_system_prompt(self, user_xp: int) -> str:
        if user_xp >= 1000:
            return (
                "You are Mayda, a warm and energetic AI host for a premium user. "
                "Be respectful, fun, concise, and socially helpful."
            )

        return (
            "You are Mayda, a friendly AI host in a social room. "
            "Encourage conversation, reduce awkward silence, and keep the tone playful."
        )

    async def translate_message(self, text: str, target_lang: str) -> str:
        if not self.model:
            return text

        prompt = (
            f"Translate this message into {target_lang}. "
            f"Keep it natural and conversational. Message: {text}"
        )

        try:
            response = await self.model.generate_content_async(prompt)
            return response.text.strip()
        except Exception:
            return text

    async def proactive_host(self, context: str, user_xp: int = 0) -> str:
        if not self.model:
            return "Mayda: Quiet room detected. Somebody start with a fun question 😄"

        sys_prompt = self.get_system_prompt(user_xp)

        prompt = (
            f"System: {sys_prompt}\n"
            f"Room context: {context}\n"
            f"Write one short, fun message to restart the conversation."
        )

        try:
            response = await self.model.generate_content_async(prompt)
            return f"Mayda: {response.text.strip()}"
        except Exception:
            return "Mayda: Server is thinking too hard. Someone say something funny 😄"


mayda_bot = MaydaAIHost()


# ==============================================================================
# CONTENT ENGINE
# ==============================================================================

class ContentEngine:
    def __init__(self):
        self.room_logs: Dict[str, List[dict]] = {}

    def log_event(self, room_id: str, event_type: str, data: str):
        if room_id not in self.room_logs:
            self.room_logs[room_id] = []

        self.room_logs[room_id].append({
            "time": time.time(),
            "type": event_type,
            "data": data,
        })

    async def generate_highlight(self, room_id: str):
        logs = self.room_logs.get(room_id, [])

        if len(logs) >= 50:
            summary = {
                "room_id": room_id,
                "event_count": len(logs),
                "created_at": time.time(),
                "status": "highlight_ready",
            }

            print(f"[ContentEngine] Highlight generated: {summary}")
            self.room_logs[room_id] = []

            return summary

        return None


content_engine = ContentEngine()


# ==============================================================================
# FRIENDLY CHALLENGE SYSTEM
# ==============================================================================

class FriendlyChallengeSystem:
    """
    No real money. No gambling.
    This is only virtual XP / credits for social gamification.
    """

    def __init__(self):
        self.challenges: Dict[str, dict] = {}

    def create_challenge(self, creator_id: str, description: str, virtual_points: int = 100) -> dict:
        challenge_id = str(uuid.uuid4())

        self.challenges[challenge_id] = {
            "id": challenge_id,
            "creator_id": creator_id,
            "description": description,
            "virtual_points": virtual_points,
            "status": "OPEN",
            "created_at": time.time(),
        }

        return self.challenges[challenge_id]

    def complete_challenge(self, challenge_id: str, winner_id: str) -> dict:
        challenge = self.challenges.get(challenge_id)

        if not challenge:
            return {
                "ok": False,
                "error": "challenge_not_found",
            }

        if challenge["status"] != "OPEN":
            return {
                "ok": False,
                "error": "challenge_already_closed",
            }

        challenge["status"] = "COMPLETED"
        challenge["winner_id"] = winner_id
        challenge["completed_at"] = time.time()

        return {
            "ok": True,
            "challenge": challenge,
        }


challenge_system = FriendlyChallengeSystem()


# ==============================================================================
# ROOM MANAGER
# ==============================================================================

class UserConnection:
    def __init__(self, websocket: WebSocket, user_id: str, interests: List[str]):
        self.websocket = websocket
        self.user_id = user_id
        self.interests = interests
        self.is_premium = False


class GlobalRoomManager:
    def __init__(self):
        self.waiting_users: List[UserConnection] = []
        self.active_rooms: Dict[str, List[UserConnection]] = {}
        self.room_last_active: Dict[str, float] = {}
        self.silence_tasks: Dict[str, asyncio.Task] = {}

    async def connect_1v1(self, user: UserConnection):
        await user.websocket.accept()

        for waiter in list(self.waiting_users):
            if set(user.interests).intersection(set(waiter.interests)):
                self.waiting_users.remove(waiter)

                room_id = str(uuid.uuid4())
                self.active_rooms[room_id] = [user, waiter]
                self.room_last_active[room_id] = time.time()

                msg = {
                    "type": "match_found",
                    "room_id": room_id,
                    "message": "Match found. You have been connected.",
                }

                await user.websocket.send_text(json.dumps(msg))
                await waiter.websocket.send_text(json.dumps(msg))

                asyncio.create_task(self.timer_1min(room_id, user, waiter))
                return

        self.waiting_users.append(user)

        await user.websocket.send_text(json.dumps({
            "type": "waiting",
            "message": "Searching for someone with similar interests.",
        }))

    async def timer_1min(self, room_id: str, u1: UserConnection, u2: UserConnection):
        await asyncio.sleep(60)

        if room_id in self.active_rooms:
            msg = json.dumps({
                "type": "session_end",
                "message": "One-minute intro session ended.",
                "room_id": room_id,
            })

            try:
                await u1.websocket.send_text(msg)
                await u2.websocket.send_text(msg)
                await u1.websocket.close()
                await u2.websocket.close()
            except Exception:
                pass

            self.active_rooms.pop(room_id, None)

    async def monitor_silence(self, room_id: str):
        while True:
            await asyncio.sleep(5)

            last_time = self.room_last_active.get(room_id, time.time())

            if time.time() - last_time > 30:
                joke = await mayda_bot.proactive_host(
                    context="The room has been silent for 30 seconds.",
                    user_xp=0,
                )

                await redis_client.publish(
                    f"room_{room_id}",
                    json.dumps({
                        "sender": "Mayda",
                        "text": joke,
                        "type": "ai_host",
                    }),
                )

                self.room_last_active[room_id] = time.time()


room_manager = GlobalRoomManager()


# ==============================================================================
# WEBSOCKET: GLOBAL CHAT
# ==============================================================================

@app.websocket("/ws/global_chat/{room_id}")
async def global_chat_endpoint(
    websocket: WebSocket,
    room_id: str,
    token: str = Query(""),
    target_lang: str = Query("mn"),
    xp: int = Query(0),
):
    user_id = verify_token(token)

    if not user_id:
        await websocket.close(code=1008, reason="Invalid token")
        return

    await websocket.accept()

    pubsub = redis_client.pubsub()
    await pubsub.subscribe(f"room_{room_id}")

    room_manager.room_last_active[room_id] = time.time()

    if room_id not in room_manager.silence_tasks:
        room_manager.silence_tasks[room_id] = asyncio.create_task(
            room_manager.monitor_silence(room_id)
        )

    async def listen_to_redis():
        try:
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue

                data = json.loads(message["data"])

                if data.get("sender") != user_id and data.get("sender") != "Mayda":
                    translated = await mayda_bot.translate_message(
                        data.get("text", ""),
                        target_lang,
                    )
                    data["translated_text"] = translated

                await websocket.send_text(json.dumps(data))

        except asyncio.CancelledError:
            pass

    listener_task = asyncio.create_task(listen_to_redis())

    try:
        await websocket.send_text(json.dumps({
            "type": "system",
            "message": "Connected to global chat.",
            "room_id": room_id,
            "user_id": user_id,
        }))

        while True:
            text_data = await websocket.receive_text()

            if not await check_rate_limit(user_id):
                await websocket.send_text(json.dumps({
                    "type": "rate_limit",
                    "message": "You are sending messages too fast.",
                }))
                continue

            room_manager.room_last_active[room_id] = time.time()
            content_engine.log_event(room_id, "chat", f"{user_id}: {text_data}")

            keywords = ["business", "startup", "ai", "trade", "money", "мөнгө", "ашиг", "стартап"]

            if any(word in text_data.lower() for word in keywords):
                hype_msg = await mayda_bot.proactive_host(
                    context=f"People are discussing ambition or business: {text_data}",
                    user_xp=xp,
                )

                await redis_client.publish(
                    f"room_{room_id}",
                    json.dumps({
                        "sender": "Mayda",
                        "text": hype_msg,
                        "type": "ai_host",
                    }),
                )

            await redis_client.publish(
                f"room_{room_id}",
                json.dumps({
                    "sender": user_id,
                    "text": text_data,
                    "type": "chat",
                    "created_at": time.time(),
                }),
            )

            await content_engine.generate_highlight(room_id)

    except WebSocketDisconnect:
        content_engine.log_event(room_id, "system", f"{user_id} disconnected.")

    finally:
        listener_task.cancel()
        await pubsub.unsubscribe(f"room_{room_id}")
        await pubsub.close()


# ==============================================================================
# WEBSOCKET: 1V1 MATCHMAKING
# ==============================================================================

@app.websocket("/ws/match")
async def match_endpoint(
    websocket: WebSocket,
    token: str = Query(""),
    interests: str = Query("general"),
):
    user_id = verify_token(token)

    if not user_id:
        await websocket.close(code=1008, reason="Invalid token")
        return

    user = UserConnection(
        websocket=websocket,
        user_id=user_id,
        interests=[x.strip() for x in interests.split(",") if x.strip()],
    )

    await room_manager.connect_1v1(user)

    try:
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        if user in room_manager.waiting_users:
            room_manager.waiting_users.remove(user)


# ==============================================================================
# WEBSOCKET: WEBRTC SIGNALING
# ==============================================================================

@app.websocket("/ws/webrtc/{room_id}")
async def webrtc_signaling(
    websocket: WebSocket,
    room_id: str,
    token: str = Query(""),
):
    user_id = verify_token(token)

    if not user_id:
        await websocket.close(code=1008, reason="Invalid token")
        return

    await websocket.accept()

    pubsub = redis_client.pubsub()
    await pubsub.subscribe(f"webrtc_{room_id}")

    async def listen_to_redis_webrtc():
        try:
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue

                data = json.loads(message["data"])

                if data.get("sender") != user_id:
                    await websocket.send_text(json.dumps(data))

        except asyncio.CancelledError:
            pass

    listener_task = asyncio.create_task(listen_to_redis_webrtc())

    try:
        while True:
            text_data = await websocket.receive_text()

            if not await check_rate_limit(user_id):
                continue

            payload = json.loads(text_data)
            payload["sender"] = user_id
            payload["created_at"] = time.time()

            await redis_client.publish(
                f"webrtc_{room_id}",
                json.dumps(payload),
            )

    except WebSocketDisconnect:
        pass

    finally:
        listener_task.cancel()
        await pubsub.unsubscribe(f"webrtc_{room_id}")
        await pubsub.close()


# ==============================================================================
# MEDIAPIPE AVATAR DETECTOR
# ==============================================================================

def create_face_detector():
    if python is None or vision is None:
        return None

    if not os.path.exists(FACE_LANDMARKER_PATH):
        return None

    try:
        base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
        )
        return vision.FaceLandmarker.create_from_options(options)

    except Exception:
        return None


detector = create_face_detector()


@app.websocket("/ws/avatar")
async def avatar_endpoint(
    websocket: WebSocket,
    token: str = Query(""),
):
    user_id = verify_token(token)

    if not user_id:
        await websocket.close(code=1008, reason="Invalid token")
        return

    await websocket.accept()

    if not detector:
        await websocket.send_text(json.dumps({
            "type": "avatar_error",
            "message": "face_landmarker.task not found or MediaPipe detector unavailable.",
        }))
        return

    try:
        while True:
            data = await websocket.receive_text()

            if "," in data:
                data = data.split(",", 1)[1]

            img_data = base64.b64decode(data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text(json.dumps({
                    "type": "avatar_error",
                    "message": "Invalid image frame.",
                }))
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = detector.detect(mp_image)

            if result.face_blendshapes:
                blendshapes = {
                    cat.category_name: float(cat.score)
                    for cat in result.face_blendshapes[0]
                }

                await websocket.send_text(json.dumps({
                    "type": "blendshapes",
                    "user_id": user_id,
                    "data": blendshapes,
                }))

    except WebSocketDisconnect:
        pass


# ==============================================================================
# WEBSOCKET: SQUAD FRIENDLY CHALLENGE
# ==============================================================================

@app.websocket("/ws/squad/{squad_id}")
async def squad_endpoint(
    websocket: WebSocket,
    squad_id: str,
    token: str = Query(""),
):
    user_id = verify_token(token)

    if not user_id:
        await websocket.close(code=1008, reason="Invalid token")
        return

    await websocket.accept()

    await websocket.send_text(json.dumps({
        "type": "squad_joined",
        "squad_id": squad_id,
        "user_id": user_id,
        "message": "Joined squad challenge room.",
    }))

    active_challenge_id: Optional[str] = None

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                cmd = json.loads(raw)
            except Exception:
                cmd = {"action": raw}

            action = cmd.get("action")

            if action == "CREATE_CHALLENGE":
                description = cmd.get("description", f"Squad challenge in {squad_id}")
                virtual_points = int(cmd.get("virtual_points", 100))

                challenge = challenge_system.create_challenge(
                    creator_id=user_id,
                    description=description,
                    virtual_points=virtual_points,
                )

                active_challenge_id = challenge["id"]

                await websocket.send_text(json.dumps({
                    "type": "challenge_created",
                    "challenge": challenge,
                }))

            elif action == "COMPLETE_CHALLENGE":
                challenge_id = cmd.get("challenge_id") or active_challenge_id

                if not challenge_id:
                    await websocket.send_text(json.dumps({
                        "type": "challenge_error",
                        "error": "No challenge selected.",
                    }))
                    continue

                result = challenge_system.complete_challenge(
                    challenge_id=challenge_id,
                    winner_id=user_id,
                )

                await websocket.send_text(json.dumps({
                    "type": "challenge_result",
                    "result": result,
                }))

            else:
                await websocket.send_text(json.dumps({
                    "type": "unknown_action",
                    "message": "Supported actions: CREATE_CHALLENGE, COMPLETE_CHALLENGE",
                }))

    except WebSocketDisconnect:
        pass
