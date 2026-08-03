from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGO_URI, MONGO_DB_NAME

client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB_NAME]

# Collections
users_collection = db["users"]
patients_collection = db["patients"]
clinical_records_collection = db["clinical_records"]
predictions_collection = db["predictions"]
feedback_collection = db["feedback"]
