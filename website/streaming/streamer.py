"""
Streaming Prediction Engine
Continuously generates predictions from last training date to today.
Uses SQLite for robust historical data storage instead of a JSON cache.
"""
import asyncio
import sqlite3
from datetime import datetime, timedelta
import json
from pathlib import Path

from predictor import get_predictor

class PredictionStreamer:
    def __init__(self):
        self.predictor = get_predictor()
        # Ensure database directory exists
        db_dir = Path(__file__).parent.parent / "database"
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / "historical.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for historical predictions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    route TEXT,
                    date TEXT,
                    predicted_price REAL,
                    confidence REAL,
                    generated_at TEXT,
                    PRIMARY KEY (route, date)
                )
            ''')
            conn.commit()
    
    def _save_to_db(self, predictions):
        """Save a list of predictions to the SQLite db"""
        generated_at = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for p in predictions:
                cursor.execute('''
                    INSERT OR REPLACE INTO predictions 
                    (route, date, predicted_price, confidence, generated_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (p['route'], p['date'], p['predicted_price'], p['confidence'], generated_at))
            conn.commit()

    def generate_stream(self, days_back=30):
        """
        Generate predictions from (today - days_back) to today.
        Returns predictions grouped by route.
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        # Don't go before last training date
        if start_date < self.predictor.last_training_date.date():
            start_date = self.predictor.last_training_date.date()
        
        predictions = self.predictor.predict_range(start_date, end_date)
        
        # Save to database
        self._save_to_db(predictions)
        
        # Group by route for return value
        by_route = {}
        for pred in predictions:
            route = pred['route']
            if route not in by_route:
                by_route[route] = []
            by_route[route].append(pred)
        
        return {
            'generated_at': datetime.now().isoformat(),
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'predictions_by_route': by_route
        }
    
    def get_cached(self):
        """Get cached predictions if available and recent (from DB)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Get the most recent generation time
            cursor.execute('SELECT MAX(generated_at) as last_gen FROM predictions')
            row = cursor.fetchone()
            if not row or not row['last_gen']:
                return None
            
            generated_at = datetime.fromisoformat(row['last_gen'])
            if (datetime.now() - generated_at).total_seconds() > 3600:  # 1 hour
                return None
            
            # Fetch all recent predictions
            cursor.execute('SELECT * FROM predictions')
            rows = cursor.fetchall()
            
            if not rows:
                return None

            by_route = {}
            start_date = "9999-99-99"
            end_date = "0000-00-00"
            for r in rows:
                route = r['route']
                if route not in by_route:
                    by_route[route] = []
                by_route[route].append({
                    'route': route,
                    'date': r['date'],
                    'predicted_price': r['predicted_price'],
                    'confidence': r['confidence']
                })
                start_date = min(start_date, r['date'])
                end_date = max(end_date, r['date'])

            return {
                'generated_at': generated_at.isoformat(),
                'start_date': start_date,
                'end_date': end_date,
                'predictions_by_route': by_route
            }
    
    def get_today_prediction(self, route=None):
        """Get today's prediction for a specific route or all routes"""
        today = datetime.now().date().isoformat()
        
        if route:
            # Check DB first
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM predictions WHERE route = ? AND date = ?', (route, today))
                row = cursor.fetchone()
                if row:
                    generated_at = datetime.fromisoformat(row['generated_at'])
                    if (datetime.now() - generated_at).total_seconds() <= 3600:
                        return {
                            'route': row['route'],
                            'date': row['date'],
                            'predicted_price': row['predicted_price'],
                            'confidence': row['confidence']
                        }
            # Generate on-demand
            pred = self.predictor.predict_single(route, datetime.now().date())
            if pred:
                self._save_to_db([pred])
            return pred
        else:
            preds = self.predictor.predict_today()
            if preds:
                self._save_to_db(preds)
            return preds


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Shipping Price Prediction Streamer")
    parser.add_argument("--days", type=int, default=30, help="Days to predict back")
    parser.add_argument("--route", type=str, help="Specific route to predict")
    parser.add_argument("--today", action="store_true", help="Only predict today")
    parser.add_argument("--cached", action="store_true", help="Show cached predictions")
    
    args = parser.parse_args()
    
    streamer = PredictionStreamer()
    
    if args.cached:
        cached = streamer.get_cached()
        if cached:
            print(json.dumps(cached, indent=2, default=str))
        else:
            print("No recent cache. Run without --cached to generate.")
    elif args.today:
        preds = streamer.get_today_prediction(args.route)
        if isinstance(preds, list):
            for p in preds:
                print(f"{p['route']}: ${p['predicted_price']:.0f} (confidence: {p['confidence']:.0%})")
        elif preds:
            print(f"{preds['route']}: ${preds['predicted_price']:.0f} (confidence: {preds['confidence']:.0%})")
    else:
        result = streamer.generate_stream(args.days)
        print(f"Generated predictions from {result['start_date']} to {result['end_date']}")
        
        total_preds = sum(len(v) for v in result['predictions_by_route'].values())
        print(f"Total predictions: {total_preds}")
        
        # Print today's predictions
        today = datetime.now().date().isoformat()
        print(f"\\n=== Today's Predictions ({today}) ===")
        for route, preds in result['predictions_by_route'].items():
            today_pred = [p for p in preds if p['date'] == today]
            if today_pred:
                p = today_pred[0]
                print(f"  {route}: ${p['predicted_price']:.0f}")
