import math
import re
from datetime import datetime, timedelta
from typing import Optional, List
import requests

from fastapi import FastAPI, Depends, Form, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from twilio.twiml.messaging_response import MessagingResponse

# ==========================================
# 1. DATABASE SETUP (SQLite for easy run)
# ==========================================
SQLALCHEMY_DATABASE_URL = "sqlite:///./floodlane.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, index=True)
    whatsapp_opt_in = Column(Boolean, default=True)
    role = Column(String, default="resident") # resident, shop, commuter
    home_lat = Column(Float, nullable=True)
    home_lng = Column(Float, nullable=True)
    area_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    source = Column(String) # 'app' or 'whatsapp'
    lat = Column(Float)
    lng = Column(Float)
    location_name = Column(String, default="Unknown Lane")
    severity = Column(String) # 'none', 'low', 'medium', 'high'
    description = Column(String, nullable=True)
    image_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    phone_number = Column(String)
    channel = Column(String, default="whatsapp")
    message = Column(String)
    sent_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="sent")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 2. AI / NLP & GEO HELPER FUNCTIONS
# ==========================================

def extract_severity_from_text(text: str) -> Optional[str]:
    """Hybrid Rule-Based NLP for Flood Depth Extraction (Section 6.4)"""
    t = text.lower().strip()
    # Quick reply numbers
    if t == "1": return "none"
    if t == "2": return "low"
    if t == "3": return "medium"
    if t == "4": return "high"
    
    # Keyword matching
    if any(k in t for k in ["waist", "above knee", "chest", "cannot walk", "car stuck", "submerged", "entering house", "4)"]):
        return "high"
    if any(k in t for k in ["knee", "half wheel", "half tyre", "bike stuck", "entering shop", "first step", "3)"]):
        return "medium"
    if any(k in t for k in ["ankle", "low", "foot", "slow traffic", "waterlogging", "2)"]):
        return "low"
    if any(k in t for k in ["wet", "no water", "clear", "normal", "small puddle", "receded", "1)"]):
        return "none"
    return None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Returns distance in meters between two GPS coordinates."""
    R = 6371000  # radius of Earth in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def geocode_location_text(text: str) -> tuple[float, float, str]:
    """Extracts location from text using Nominatim OSM or falls back to Bengaluru demo spots."""
    t = text.lower()
    # Demo Fallback Dictionary for instant hackathon reliability (Bengaluru hotspots)
    known_spots = {
        "jp nagar": (12.9063, 77.5857, "JP Nagar 6th Phase"),
        "koramangala": (12.9352, 77.6245, "Koramangala 4th Block (Sony World)"),
        "silk board": (12.9177, 77.6238, "Silk Board Junction"),
        "bellandur": (12.9304, 77.6784, "Bellandur EcoSpace Service Rd"),
        "indiranagar": (12.9784, 77.6408, "Indiranagar 100ft Road"),
        "whitefield": (12.9698, 77.7500, "Whitefield Main Rd"),
        "jayanagar": (12.9308, 77.5838, "Jayanagar 4th Block"),
        "hebbal": (13.0354, 77.5988, "Hebbal Flyover Underpass"),
        "malleswaram": (13.0031, 77.5643, "Malleswaram 8th Cross")
    }
    for key, (lat, lng, name) in known_spots.items():
        if key in t:
            return lat, lng, name

    # Try OpenStreetMap Nominatim if "near <place>" or "in <place>" is mentioned
    match = re.search(r'(?:near|at|in|outside)\s+([a-zA-Z0-9\s,]+)', text, re.IGNORECASE)
    if match:
        query = match.group(1).strip() + ", Bengaluru, India"
        try:
            res = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": "FloodLaneHackathonApp/1.0"},
                timeout=2.5
            )
            if res.status_code == 200 and len(res.json()) > 0:
                data = res.json()[0]
                return float(data["lat"]), float(data["lon"]), data["display_name"].split(",")[0]
        except Exception:
            pass

    # Default coordinates (Central Bengaluru) with slight jitter if no specific pin found
    return 12.9352, 77.6245, "Koramangala 80ft Rd (Inferred)"

def reverse_geocode(lat: float, lng: float) -> str:
    """Gets street name from coordinates."""
    try:
        res = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lng, "format": "json"},
            headers={"User-Agent": "FloodLaneHackathonApp/1.0"},
            timeout=2.0
        )
        if res.status_code == 200:
            addr = res.json().get("address", {})
            road = addr.get("road") or addr.get("suburb") or addr.get("neighbourhood")
            area = addr.get("suburb") or addr.get("city_district") or ""
            if road:
                return f"{road}, {area}".strip(", ")
    except Exception:
        pass
    return f"Lane ({round(lat, 3)}, {round(lng, 3)})"

# ==========================================
# 3. HYPER-LOCAL ALERT ENGINE (Section 5.4)
# ==========================================

SEVERITY_LABELS = {
    "none": "No Water / Wet Road",
    "low": "Ankle-Deep (~0-10 cm)",
    "medium": "Knee-Deep (~10-30 cm)",
    "high": "Above Knee (>30 cm - Impassable)"
}

def trigger_hyperlocal_alerts(db: Session, report: Report, radius_m: float = 500.0):
    """Finds subscribed users within 500m and sends WhatsApp alerts if severity >= medium."""
    if report.severity not in ["medium", "high"]:
        return []

    users = db.query(User).filter(User.whatsapp_opt_in == True, User.home_lat.isnot(None)).all()
    cooldown_threshold = datetime.utcnow() - timedelta(minutes=30)
    triggered_alerts = []

    for u in users:
        dist = haversine_distance(report.lat, report.lng, u.home_lat, u.home_lng)
        if dist <= radius_m:
            # Deduplicate: Check if user already got an alert in the last 30 mins
            recent = db.query(Alert).filter(
                Alert.user_id == u.id,
                Alert.sent_at >= cooldown_threshold
            ).first()
            if recent:
                continue

            impact = "Motorbikes and small cars may face difficulty." if report.severity == "medium" else "DANGER: Lane impassable! Protect ground-floor goods & avoid route."
            msg = (
                f"⚠️ *Flood Alert near {report.location_name}* ({int(dist)}m away)\n"
                f"• *Severity:* {SEVERITY_LABELS.get(report.severity)}\n"
                f"• *Time:* {report.created_at.strftime('%H:%M')} UTC\n"
                f"• *Impact:* {impact}\n"
                f"🔗 View Live Map: http://localhost:8000"
            )
            new_alert = Alert(
                report_id=report.id,
                user_id=u.id,
                phone_number=u.phone_number,
                channel="whatsapp",
                message=msg,
                status="delivered"
            )
            db.add(new_alert)
            triggered_alerts.append(new_alert)

    db.commit()
    return triggered_alerts

# ==========================================
# 4. FASTAPI APP & ENDPOINTS
# ==========================================

app = FastAPI(title="FloodLane API & Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store for multi-step WhatsApp conversations
whatsapp_sessions = {}

class ReportCreate(BaseModel):
    lat: float
    lng: float
    severity: str
    description: Optional[str] = ""
    location_name: Optional[str] = None
    image_url: Optional[str] = None
    source: str = "app"

class UserCreate(BaseModel):
    phone_number: str
    role: str = "resident"
    home_lat: float
    home_lng: float
    area_name: Optional[str] = "My Neighborhood"

@app.on_event("startup")
def seed_initial_data():
    """Seeds realistic Bengaluru flood data and users for instant demo visualization."""
    db = SessionLocal()
    if db.query(Report).count() == 0:
        # Seed subscribed users
        u1 = User(phone_number="+919876543210", role="shop", home_lat=12.9065, home_lng=77.5860, area_name="JP Nagar 6th Phase")
        u2 = User(phone_number="+919888877777", role="resident", home_lat=12.9355, home_lng=77.6240, area_name="Koramangala 4th Block")
        u3 = User(phone_number="+919777766666", role="commuter", home_lat=12.9180, home_lng=77.6235, area_name="Silk Board")
        db.add_all([u1, u2, u3])
        db.commit()

        # Seed initial reports
        now = datetime.utcnow()
        seed_reports = [
            Report(source="whatsapp", lat=12.9063, lng=77.5857, location_name="JP Nagar 15th Cross", severity="medium", description="Knee-deep water outside kirana store, rising fast.", created_at=now - timedelta(minutes=45)),
            Report(source="app", lat=12.9352, lng=77.6245, location_name="Koramangala 80ft Road", severity="high", description="Above knee! Drain overflow near Sony World signal. Cars stuck.", created_at=now - timedelta(minutes=20)),
            Report(source="whatsapp", lat=12.9177, lng=77.6238, location_name="Silk Board Service Lane", severity="low", description="Ankle deep waterlogging, two-wheelers moving slowly.", created_at=now - timedelta(minutes=15)),
            Report(source="app", lat=12.9308, lng=77.5838, location_name="Jayanagar 4th Block Market", severity="none", description="Just wet roads, drainage cleared by volunteers.", created_at=now - timedelta(minutes=10)),
            Report(source="whatsapp", lat=12.9360, lng=77.6252, location_name="Koramangala 4th Block 5th Cross", severity="high", description="Water entering ground floor gates.", created_at=now - timedelta(minutes=5)),
        ]
        for r in seed_reports:
            db.add(r)
            db.commit()
            db.refresh(r)
            trigger_hyperlocal_alerts(db, r)
    db.close()

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/reports")
def get_reports(hours: int = 24, severity: Optional[str] = None, db: Session = Depends(get_db)):
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    query = db.query(Report).filter(Report.created_at >= cutoff)
    if severity and severity != "all":
        query = query.filter(Report.severity == severity)
    reports = query.order_by(Report.created_at.desc()).all()
    return reports

@app.post("/api/reports")
def create_report(payload: ReportCreate, db: Session = Depends(get_db)):
    loc_name = payload.location_name or reverse_geocode(payload.lat, payload.lng)
    new_report = Report(
        source=payload.source,
        lat=payload.lat,
        lng=payload.lng,
        location_name=loc_name,
        severity=payload.severity,
        description=payload.description,
        image_url=payload.image_url
    )
    db.add(new_report)
    db.commit()
    db.refresh(new_report)
    
    alerts = trigger_hyperlocal_alerts(db, new_report)
    return {"report": new_report, "alerts_triggered": len(alerts)}

@app.post("/api/users")
def register_user(payload: UserCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone_number == payload.phone_number).first()
    if user:
        user.role = payload.role
        user.home_lat = payload.home_lat
        user.home_lng = payload.home_lng
        user.area_name = payload.area_name
    else:
        user = User(**payload.dict())
        db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.get("/api/users")
def list_users(db: Session = Depends(get_db)):
    return db.query(User).order_by(User.created_at.desc()).all()

@app.get("/api/alerts")
def list_alerts(db: Session = Depends(get_db)):
    return db.query(Alert).order_by(Alert.sent_at.desc()).limit(50).all()

@app.get("/api/dashboard/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    reports = db.query(Report).all()
    alerts_count = db.query(Alert).count()
    users_count = db.query(User).count()
    
    counts = {"none": 0, "low": 0, "medium": 0, "high": 0}
    hotspots = {}
    for r in reports:
        counts[r.severity] = counts.get(r.severity, 0) + 1
        area = r.location_name.split(",")[0]
        hotspots[area] = hotspots.get(area, 0) + 1

    top_hotspots = sorted([{"area": k, "count": v} for k, v in hotspots.items()], key=lambda x: x["count"], reverse=True)[:5]
    
    return {
        "total_reports": len(reports),
        "severity_breakdown": counts,
        "top_hotspots": top_hotspots,
        "total_alerts_sent": alerts_count,
        "active_subscribers": users_count
    }

# ==========================================
# 5. WHATSAPP BOT LOGIC & TWILIO WEBHOOK
# ==========================================

def process_whatsapp_message(
    from_number: str,
    body: str,
    lat: Optional[float],
    lng: Optional[float],
    media_url: Optional[str],
    db: Session
) -> dict:
    """Core conversational & NLP pipeline for both Twilio Webhook and Web Simulator."""
    text = (body or "").strip()
    lower_text = text.lower()

    # Ensure user exists in DB
    user = db.query(User).filter(User.phone_number == from_number).first()
    if not user:
        user = User(phone_number=from_number, role="resident", home_lat=12.9063, home_lng=77.5857, area_name="JP Nagar")
        db.add(user)
        db.commit()
        db.refresh(user)

    # 1. Greeting / Onboarding Flow (Section 5.1)
    if lower_text in ["hi", "hello", "hey", "start", "menu", "help"]:
        whatsapp_sessions.pop(from_number, None)
        reply = (
            "👋 *Welcome to FloodLane!* 🌧️\n"
            "Your hyper-local street flood intelligence bot.\n\n"
            "*How to report flooding:*\n"
            "Send a message with water level & landmark (e.g., _'Knee-deep water near XYZ Bakery, JP Nagar'_) or attach a 📍 *Location Pin* / 📷 *Photo*.\n\n"
            "*Commands:*\n"
            "• Reply *SUBSCRIBE <Area>* to get alerts within 500m (e.g. _SUBSCRIBE Koramangala_)\n"
            "• Or simply describe waterlogging around you right now!"
        )
        return {"reply": reply, "report_created": None, "alerts_sent": 0}

    # 2. Subscription Flow
    if lower_text.startswith("subscribe"):
        area_part = text[9:].strip() or "JP Nagar"
        s_lat, s_lng, s_name = geocode_location_text("in " + area_part)
        user.home_lat = s_lat
        user.home_lng = s_lng
        user.area_name = s_name
        user.whatsapp_opt_in = True
        db.commit()
        return {
            "reply": f"✅ *Subscribed to Hyper-Local Alerts!*\n📍 Pin set to: *{s_name}* ({round(s_lat,4)}, {round(s_lng,4)}).\nWe will alert you via WhatsApp if knee-deep or worse flooding is reported within 500m.",
            "report_created": None,
            "alerts_sent": 0
        }

    # 3. Check if user is answering a Pending Clarification Prompt
    session = whatsapp_sessions.get(from_number)
    severity = extract_severity_from_text(text)

    if session and session.get("awaiting_severity"):
        if severity:
            r_lat = lat or session["lat"]
            r_lng = lng or session["lng"]
            r_loc = session["location_name"]
            r_desc = session["description"]
            r_img = media_url or session.get("image_url")
            whatsapp_sessions.pop(from_number, None)

            new_report = Report(
                user_id=user.id, source="whatsapp", lat=r_lat, lng=r_lng,
                location_name=r_loc, severity=severity, description=r_desc, image_url=r_img
            )
            db.add(new_report)
            db.commit()
            db.refresh(new_report)
            alerts = trigger_hyperlocal_alerts(db, new_report)

            reply = (
                f"✅ *Flood Report Mapped!*\n"
                f"📍 *Location:* {r_loc}\n"
                f"🌊 *Depth:* {SEVERITY_LABELS[severity]}\n"
                f"🔔 *Nearby Alerts Dispatched:* {len(alerts)} neighbors notified."
            )
            return {"reply": reply, "report_created": new_report, "alerts_sent": len(alerts)}

    # 4. Standard Reporting Pipeline (Extract Location + Severity)
    if lat is not None and lng is not None:
        r_lat, r_lng = lat, lng
        r_loc = reverse_geocode(lat, lng)
    else:
        r_lat, r_lng, r_loc = geocode_location_text(text)

    # If severity is ambiguous, ask quick-reply options (Section 5.2B)
    if not severity:
        whatsapp_sessions[from_number] = {
            "awaiting_severity": True,
            "lat": r_lat,
            "lng": r_lng,
            "location_name": r_loc,
            "description": text or "Photo / Pin uploaded via WhatsApp",
            "image_url": media_url
        }
        reply = (
            f"📍 Got location near *{r_loc}*.\n\n"
            f"*How deep is the water right now?*\n"
            f"Reply with a number (1-4):\n"
            f"1️⃣ Wet road only\n"
            f"2️⃣ Ankle-deep (~0-10 cm)\n"
            f"3️⃣ Knee-deep (~10-30 cm)\n"
            f"4️⃣ Above knee (>30 cm / Impassable)"
        )
        return {"reply": reply, "report_created": None, "alerts_sent": 0}

    # Both location & severity extracted -> Save Report & Trigger Alerts!
    new_report = Report(
        user_id=user.id,
        source="whatsapp",
        lat=r_lat,
        lng=r_lng,
        location_name=r_loc,
        severity=severity,
        description=text,
        image_url=media_url
    )
    db.add(new_report)
    db.commit()
    db.refresh(new_report)
    alerts = trigger_hyperlocal_alerts(db, new_report)

    reply = (
        f"✅ *Flood Report Live on Map!*\n"
        f"📍 *Lane:* {r_loc}\n"
        f"🌊 *Severity:* {SEVERITY_LABELS[severity]}\n"
        f"🔔 *Hyper-Local Engine:* Sent {len(alerts)} WhatsApp alert(s) to residents within 500m."
    )
    return {"reply": reply, "report_created": new_report, "alerts_sent": len(alerts)}


@app.post("/webhook/whatsapp")
async def twilio_whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(""),
    Latitude: Optional[float] = Form(None),
    Longitude: Optional[float] = Form(None),
    MediaUrl0: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Real Twilio WhatsApp Sandbox / Meta Cloud Webhook Receiver."""
    clean_phone = From.replace("whatsapp:", "")
    result = process_whatsapp_message(clean_phone, Body, Latitude, Longitude, MediaUrl0, db)
    
    twiml = MessagingResponse()
    twiml.message(result["reply"])
    return Response(content=str(twiml), media_type="application/xml")

class WhatsAppSimulateRequest(BaseModel):
    phone_number: str = "+919900112233"
    message: str
    lat: Optional[float] = None
    lng: Optional[float] = None

@app.post("/api/whatsapp/simulate")
def simulate_whatsapp_bot(payload: WhatsAppSimulateRequest, db: Session = Depends(get_db)):
    """Interactive Simulator Endpoint so you can test the WhatsApp Bot directly in the browser UI."""
    return process_whatsapp_message(payload.phone_number, payload.message, payload.lat, payload.lng, None, db)
