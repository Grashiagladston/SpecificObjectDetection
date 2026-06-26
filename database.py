import psycopg2
import numpy as np
from datetime import datetime

# ============================================
# DATABASE CONNECTION
# ============================================
try:
    import config
    
    def get_connection():
        return psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            database=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD
        )
    
    conn = get_connection()

except Exception as e:
    print(f"Connection error: {e}")
    raise e


# ============================================
# DROP OLD TABLES (One time use)
# ============================================
def drop_old_tables():
    """Drop old tables completely"""
    cur = conn.cursor()
    try:
        old_tables = [
            'crop_embeddings',
            'detection_runs',
            'detected_crops',
            'image_embeddings',
            'crops'
        ]
        
        for table in old_tables:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
        
        conn.commit()
        print("✅ Old tables dropped")
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()


# ============================================
# INIT DB - NORMAL ARRAYS (No pgvector needed)
# ============================================
def init_db():
    """Initialize DB using normal FLOAT[] instead of VECTOR"""
    cur = conn.cursor()
    
    try:
        # Detection runs table - using FLOAT[]
        cur.execute("""
            CREATE TABLE IF NOT EXISTS detection_runs (
                image_id VARCHAR(50) PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                original_image_name VARCHAR(255),
                target_object VARCHAR(100),
                detected_count INTEGER,
                full_image_embedding FLOAT[]
            );
        """)
        
        # Crop embeddings table - using FLOAT[]
        cur.execute("""
            CREATE TABLE IF NOT EXISTS crop_embeddings (
                id SERIAL PRIMARY KEY,
                image_id VARCHAR(50) REFERENCES detection_runs(image_id) ON DELETE CASCADE,
                crop_label VARCHAR(100),
                crop_embedding FLOAT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.commit()
        print("✅ Database initialized (No pgvector needed)!")
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()


# ============================================
# COSINE SIMILARITY FUNCTION (Python math)
# ============================================
def calculate_cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings in Python"""
    if emb1 is None or emb2 is None:
        return 0.0
    
    # Convert database arrays to numpy arrays
    a = np.array(emb1, dtype=np.float64)
    b = np.array(emb2, dtype=np.float64)
    
    # Calculate cosine similarity
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return float(dot_product / (norm_a * norm_b))


# ============================================
# SAVE ONLY ID + EMBEDDINGS (NO IMAGES)
# ============================================
def save_embeddings_only(image_id, original_image_name, target_object, 
                         detected_count, full_image_embedding, crop_embeddings):
    """Save only image ID and embeddings - NO image data"""
    cur = conn.cursor()
    
    try:
        cur.execute("""
            INSERT INTO detection_runs (image_id, original_image_name, target_object, 
                                        detected_count, full_image_embedding)
            VALUES (%s, %s, %s, %s, %s)
        """, (image_id, original_image_name, target_object, detected_count, list(full_image_embedding)))
        
        for crop_emb_data in crop_embeddings:
            cur.execute("""
                INSERT INTO crop_embeddings (image_id, crop_label, crop_embedding)
                VALUES (%s, %s, %s)
            """, (image_id, crop_emb_data['label'], list(crop_emb_data['embedding'])))
        
        conn.commit()
        print(f"✅ Saved embeddings for image_id: {image_id}")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error saving: {e}")
        raise e
    finally:
        cur.close()


# ============================================
# FIND SIMILAR IMAGES (Python based search)
# ============================================
def find_similar_images(embedding, threshold=0.85):
    """Find similar images using Python cosine similarity"""
    cur = conn.cursor()
    
    try:
        # Fetch ALL embeddings from database
        cur.execute("""
            SELECT image_id, timestamp, original_image_name, target_object, 
                   detected_count, full_image_embedding
            FROM detection_runs
        """)
        
        all_rows = cur.fetchall()
        results = []
        
        # Calculate similarity in Python
        for row in all_rows:
            db_emb = row[5] # full_image_embedding is at index 5
            if db_emb:
                similarity = calculate_cosine_similarity(embedding, db_emb)
                
                # Check if it meets the threshold
                if similarity >= threshold:
                    db_row = (row[0], row[1], row[2], row[3], row[4], row[5])
                    results.append((db_row, similarity))
        
        # Sort by highest similarity first
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 5 matches
        return results[:5]
        
    except Exception as e:
        print(f"❌ Error finding similar: {e}")
        raise e
    finally:
        cur.close()


# ============================================
# GET HISTORY (Only IDs + Embeddings)
# ============================================
def get_history_records():
    """Get all detection records - NO image data"""
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT image_id, timestamp, original_image_name, target_object, 
                   detected_count, full_image_embedding
            FROM detection_runs
            ORDER BY timestamp DESC
        """)
        
        results = []
        for row in cur.fetchall():
            image_id = row[0]
            
            cur2 = conn.cursor()
            cur2.execute("""
                SELECT crop_label, crop_embedding
                FROM crop_embeddings
                WHERE image_id = %s
            """, (image_id,))
            
            crop_embs = []
            for crop_row in cur2.fetchall():
                crop_embs.append({
                    'label': crop_row[0],
                    'embedding': crop_row[1]
                })
            cur2.close()
            
            results.append((row[0], row[1], row[2], row[3], row[4], row[5], crop_embs))
        
        return results
        
    except Exception as e:
        print(f"❌ Error getting history: {e}")
        raise e
    finally:
        cur.close()


# ============================================
# CLEAR ALL HISTORY
# ============================================
def clear_all_history():
    """Clear all records"""
    cur = conn.cursor()
    
    try:
        cur.execute("DELETE FROM crop_embeddings")
        cur.execute("DELETE FROM detection_runs")
        conn.commit()
        print("✅ All history cleared!")
    except Exception as e:
        conn.rollback()
        print(f"❌ Error clearing: {e}")
        raise e
    finally:
        cur.close()