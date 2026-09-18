from pathlib import Path
import shutil
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import init_db,list_scenes,save_feedback,list_feedback
from .schemas import TextSearchRequest,FeedbackRequest,ChangeRequest,TimelineRequest,TimelinePairRequest,ClusterRequest
from .services.retrieval import get_retriever
from .services.ingestion import ingest_scene
from .services.temporal import TemporalAnalyzer
from .services.timeline import earliest_supported_observation
from .services.temporal_pairing import pair_observations
from .services.clustering import EmbeddingClusterer
from .services.visualize import change_heatmap
from .services.audit import audit_event

app=FastAPI(title="SIH26227 â€” Earth Observation Intelligence Platform",version="5.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
settings.ensure_dirs()

TILES_DIR = settings.ROOT / "indexes" / "real_remoteclip" / "tiles"
TILES_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/tiles",
    StaticFiles(directory=str(TILES_DIR)),
    name="tiles"
)

app.mount(
    "/artifacts",
    StaticFiles(directory=str(settings.TMP_DIR)),
    name="artifacts"
)

@app.on_event("startup")
def startup(): init_db()

@app.get("/")
def root(): return {"project":"SIH26227","version":"5.0","offline_runtime":settings.OFFLINE_ONLY}
@app.get("/discovery/clusters")
def discovery_clusters():
    import json

    clusters_file = (
        settings.ROOT
        / "indexes"
        / "real_remoteclip"
        / "clusters.json"
    )

    if not clusters_file.exists():
        return {
            "status": "error",
            "message": "Discovery clusters have not been built yet."
        }

    with open(clusters_file, "r", encoding="utf-8") as f:
        return json.load(f)
@app.get("/health")
def health():
    return {
        "status": "ok",
        "project": "SIH26227",
        "service": "TerraVision Backend",
        "offline_runtime": settings.OFFLINE_ONLY,
        "change_backend": settings.CHANGE_BACKEND,
        "change_model_staged": bool(
            settings.CHANGE_MODEL_PATH
            and Path(settings.CHANGE_MODEL_PATH).exists()
        )
    }
@app.get("/scenes")
def scenes(): return list_scenes()

@app.post("/ingest")
async def ingest(file:UploadFile=File(...)):
    if Path(file.filename or "").suffix.lower() not in {".tif",".tiff"}: raise HTTPException(400,"GeoTIFF/COG only")
    target=settings.SCENES_DIR/Path(file.filename).name
    with target.open("wb") as f: shutil.copyfileobj(file.file,f)
    return ingest_scene(target)

@app.post("/search/text")
def text_search(req:TextSearchRequest):
    return get_retriever().search_text(req.query,req.top_k,req.sensor,req.start_date,req.end_date,req.min_quality,getattr(req,'aoi_bbox',None))

@app.post("/search/image")
async def image_search(file:UploadFile=File(...),top_k:int=10):
    target=settings.TMP_DIR/Path(file.filename or "query.png").name
    with target.open("wb") as f: shutil.copyfileobj(file.file,f)
    return get_retriever().search_image(target,top_k)

@app.post("/change-detection/upload")
async def change_upload(before_file:UploadFile=File(...), after_file:UploadFile=File(...)):
    allowed={".tif",".tiff"}
    bname=Path(before_file.filename or "before.tif").name
    aname=Path(after_file.filename or "after.tif").name
    if Path(bname).suffix.lower() not in allowed or Path(aname).suffix.lower() not in allowed:
        raise HTTPException(400,"Only GeoTIFF/COG (.tif/.tiff) files are supported")
    btarget=settings.TMP_DIR/f"upload_before_{bname}"
    atarget=settings.TMP_DIR/f"upload_after_{aname}"
    with btarget.open("wb") as f: shutil.copyfileobj(before_file.file,f)
    with atarget.open("wb") as f: shutil.copyfileobj(after_file.file,f)
    result=TemporalAnalyzer().analyze(str(btarget),str(atarget))
    # Remove internal numpy arrays before returning JSON.
    result.pop("mask_array",None); result.pop("probability_array",None)
    try:
        heat=change_heatmap(str(btarget),str(atarget))
        result["heatmap_url"]=f"/artifacts/{heat.name}"
    except Exception as e:
        result["visualization_warning"]=str(e)
    result["audit"]=audit_event("change_detection_upload",before=bname,after=aname)
    return result

@app.post("/change-detection")
def change(req:ChangeRequest):
    result=TemporalAnalyzer().analyze(req.before_path,req.after_path)
    result.pop("mask_array",None); result.pop("probability_array",None)
    try:
        heat=change_heatmap(req.before_path,req.after_path)
        result['heatmap_url']=f'/artifacts/{heat.name}'
    except Exception as e: result['visualization_warning']=str(e)
    result['audit']=audit_event('change_detection',before=req.before_path,after=req.after_path)
    return result

@app.post("/timeline")
def timeline(req:TimelineRequest): return earliest_supported_observation(req.observations,TemporalAnalyzer(),req.min_confidence)

@app.post("/timeline/pairs")
def timeline_pairs(req:TimelinePairRequest): return pair_observations([x.model_dump() for x in req.observations],req.max_gap_days)

@app.post("/clusters")
def clusters(req:ClusterRequest): return EmbeddingClusterer().run(req.n_clusters,req.top_k)

@app.post("/analyst/feedback")
def feedback(req:FeedbackRequest):
    result=save_feedback(req.model_dump()); result['audit']=audit_event('analyst_feedback',decision=req.decision,tile_id=req.tile_id); return result

@app.get("/analyst/audit")
def analyst_audit(limit:int=100):
    limit=max(1,min(limit,500))
    return {"count":len(list_feedback(limit)),"records":list_feedback(limit)}
@app.get("/system/capabilities")
def capabilities():
    return {
        "semantic_retrieval":True,"image_to_image":True,"temporal_change_analysis":True,
        "registration":True,"quality_scoring":True,"incremental_ingestion":True,
        "provenance":True,"offline_runtime":True,"aoi_filtering":True,
        "timeline_pairing":True,"embedding_clustering":True,"change_visualization":True,
        "remote_sensing_vlm_adapter":True,
        "remoteclip_local_backend": settings.MODEL_BACKEND == "remoteclip",
        "real_sentinel2_staged": settings.REAL_DATA_DIR.exists() and any(settings.REAL_DATA_DIR.rglob("*.tif")),
        "learned_change_backend": settings.CHANGE_BACKEND,
        "learned_change_model_staged": bool(settings.CHANGE_MODEL_PATH and Path(settings.CHANGE_MODEL_PATH).exists()),
        "note":"Real mode requires locally staged Sentinel-2 L2A imagery, a locally packaged RemoteCLIP checkpoint, and a separately validated change-detection checkpoint. Demo mode is never presented as a real accuracy result."
    }

@app.get("/evaluation")
def evaluation():
    return {"status":"ready_for_held_out_evaluation","metrics":["Precision@K","Recall@K","mAP","MRR","F1","IoU","query_latency","index_build_time","incremental_ingest_time","storage_footprint"],"warning":"No performance number is claimed until measured on held-out SIH-style labels."}


