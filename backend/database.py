from sqlalchemy import create_engine, Column, Integer, \
    String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

# SQLite database — no server needed
DATABASE_URL = "sqlite:///./rare_disease_predictions.db"

engine        = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal  = sessionmaker(
    autocommit=False, autoflush=False, bind=engine)
Base          = declarative_base()


class PredictionRecord(Base):
    """Stores every prediction made by the system"""
    __tablename__ = "predictions"

    id            = Column(Integer, primary_key=True,
                           index=True)
    timestamp     = Column(DateTime,
                           default=datetime.utcnow)
    symptoms      = Column(Text)
    has_image     = Column(String(10), default="No")
    top1_disease  = Column(String(200))
    top1_prob     = Column(Float)
    top3_diseases = Column(Text)  # JSON string
    top5_diseases = Column(Text)  # JSON string
    top_k         = Column(Integer, default=5)
    model_used    = Column(String(50),
                           default="fusion")


class SymptomAnalysis(Base):
    """Stores symptom frequency analysis"""
    __tablename__ = "symptom_analysis"

    id           = Column(Integer, primary_key=True,
                          index=True)
    timestamp    = Column(DateTime,
                          default=datetime.utcnow)
    symptom      = Column(String(200))
    frequency    = Column(Integer, default=1)


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(db, symptoms, predictions,
                    has_image=False, top_k=5):
    """Save a prediction to database"""
    top1 = predictions[0] if predictions else {}
    top3 = predictions[:3] if predictions else []
    top5 = predictions[:5] if predictions else []

    record = PredictionRecord(
        symptoms      = symptoms if isinstance(
                            symptoms, str)
                        else ', '.join(symptoms),
        has_image     = "Yes" if has_image else "No",
        top1_disease  = top1.get('disease', ''),
        top1_prob     = top1.get('probability', 0),
        top3_diseases = json.dumps(top3),
        top5_diseases = json.dumps(top5),
        top_k         = top_k
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_analytics(db):
    """Get prediction analytics"""
    from sqlalchemy import func

    total = db.query(PredictionRecord).count()

    # Top predicted diseases
    top_diseases = db.query(
        PredictionRecord.top1_disease,
        func.count(PredictionRecord.top1_disease
                   ).label('count')
    ).group_by(
        PredictionRecord.top1_disease
    ).order_by(
        func.count(
            PredictionRecord.top1_disease).desc()
    ).limit(10).all()

    # Average confidence
    avg_conf = db.query(
        func.avg(PredictionRecord.top1_prob)
    ).scalar() or 0

    return {
        "total_predictions" : total,
        "avg_confidence"    : round(avg_conf, 2),
        "top_predicted"     : [
            {"disease": d, "count": c}
            for d, c in top_diseases
        ]
    }