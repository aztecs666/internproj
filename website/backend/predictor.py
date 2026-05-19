"""
Shipping Price Prediction Engine
Loads trained XGBoost model and generates predictions for dates from 
last training date to today.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class ShippingPredictor:
    def __init__(self):
        self.model_path = Path(r"D:\Internproj\selectedmodel\xgb_80_20_lr001_depth5.json")
        self.data_path = Path(r"D:\Internproj\new dataset\engineered_ml_dataset.csv")
        self.model = None
        self.feature_names = None
        self.historical_data = None
        self.last_training_date = None
        self.routes = [
            'XSICFENE_FarEast_NorthEurope',
            'XSICNEFE_NorthEurope_FarEast',
            'XSICFEUW_FarEast_USWestCoast',
            'XSICUWFE_USWestCoast_FarEast'
        ]
        self._load_model()
        self._load_data()
    
    def _load_model(self):
        """Load the trained XGBoost model"""
        self.model = xgb.Booster()
        self.model.load_model(str(self.model_path))
        self.feature_names = self.model.feature_names
        print(f"Model loaded: {len(self.feature_names)} features")
    
    def _load_data(self):
        """Load and prepare historical data"""
        self.historical_data = pd.read_csv(self.data_path)
        self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'])
        self.last_training_date = self.historical_data['Date'].max()
        
        # One-hot encode routes if not already done
        if 'Route_XSICFENE_FarEast_NorthEurope' not in self.historical_data.columns:
            self.historical_data = pd.get_dummies(
                self.historical_data, 
                columns=['Route'], 
                prefix='Route'
            )
            # Ensure all route columns exist
            for route in self.routes:
                col = f'Route_{route}'
                if col not in self.historical_data.columns:
                    self.historical_data[col] = 0
        
        print(f"Data loaded: {len(self.historical_data)} rows, last date: {self.last_training_date.date()}")
    
    def _get_template_row(self, route, base_date):
        """Get a template row for prediction based on historical patterns"""
        # Find the same day of week in historical data for weather/indices
        day_of_week = base_date.weekday() if hasattr(base_date, 'weekday') else base_date.isoweekday() - 1
        
        # Get a representative row from historical data for this route
        route_data = self.historical_data[
            self.historical_data[f'Route_{route}'] == 1
        ].copy()
        
        if len(route_data) == 0:
            return None
        
        # Use the most recent available data as base
        template = route_data.iloc[-1:].copy()
        
        # Update date-dependent features
        template['Month'] = base_date.month
        template['Week_of_Year'] = base_date.isocalendar()[1]
        template['Is_Peak_Season'] = 1 if base_date.month in [8, 9, 10, 11, 12] else 0
        
        return template
    
    def _calculate_lag_features(self, route, prediction_date):
        """Calculate lag features based on historical prices"""
        route_data = self.historical_data[
            self.historical_data[f'Route_{route}'] == 1
        ].sort_values('Date')
        
        # Get price map with Timestamp keys
        price_map = route_data.set_index('Date')['Price_USD'].to_dict()
        
        # Convert prediction_date to Timestamp
        pred_ts = pd.Timestamp(prediction_date)
        
        # Calculate lag dates
        lag_7_date = pred_ts - timedelta(days=7)
        lag_30_date = pred_ts - timedelta(days=30)
        
        # Find closest available prices
        lag_7_price = self._find_closest_price(price_map, lag_7_date)
        lag_30_price = self._find_closest_price(price_map, lag_30_date)
        
        # Calculate momentum
        lag_14_date = pred_ts - timedelta(days=14)
        lag_14_price = self._find_closest_price(price_map, lag_14_date)
        
        momentum_7d = (lag_7_price - lag_14_price) / lag_14_price if lag_14_price > 0 else 0
        momentum_30d = (lag_7_price - lag_30_price) / lag_30_price if lag_30_price > 0 else 0
        
        # Calculate volatility (rolling std of last 7 days)
        recent_prices = []
        for i in range(7):
            d = pred_ts - timedelta(days=i)
            p = self._find_closest_price(price_map, d)
            if p is not None:
                recent_prices.append(p)
        
        volatility_7d = np.std(recent_prices) / np.mean(recent_prices) if len(recent_prices) > 1 and np.mean(recent_prices) > 0 else 0.03
        
        # Z-score
        all_prices = list(price_map.values())
        zscore = (lag_7_price - np.mean(all_prices)) / np.std(all_prices) if np.std(all_prices) > 0 else 0
        
        return {
            'Price_Lag_7d': lag_7_price,
            'Price_Lag_30d': lag_30_price,
            'Price_Momentum_7d': momentum_7d,
            'Price_Momentum_30d': momentum_30d,
            'Price_Volatility_7d': volatility_7d,
            'Price_Volatility_30d': volatility_7d * 1.2,
            'Price_Zscore_30d': zscore
        }
    
    def _find_closest_price(self, price_map, target_date):
        """Find the closest available price for a given date"""
        # Ensure target_date is a Timestamp for comparison
        target_ts = pd.Timestamp(target_date)
        
        if target_ts in price_map:
            return price_map[target_ts]
        
        # Find closest date
        dates = sorted(price_map.keys())
        closest = min(dates, key=lambda d: abs((d - target_ts).days))
        return price_map[closest]
    
    def _get_cyclone_features(self, route, prediction_date):
        """Get cyclone features - these change slowly, use recent values"""
        route_data = self.historical_data[
            self.historical_data[f'Route_{route}'] == 1
        ].sort_values('Date')
        
        if len(route_data) == 0:
            return {}
        
        # Use recent cyclone patterns
        recent = route_data.tail(30)
        
        return {
            'Route_Max_cyclone_wind': recent['Route_Max_cyclone_wind'].mean(),
            'Route_Min_cyclone_slp': recent['Route_Min_cyclone_slp'].mean(),
            'Route_Min_cyclone_dist': recent['Route_Min_cyclone_dist'].mean(),
            'Route_cyclone_active': recent['Route_cyclone_active'].mean(),
            'Rolling_7d_Cyclone_Days': 3.0,  # Average
            'Cyclone_Wind_Lag7d': recent['Route_Max_cyclone_wind'].iloc[-1],
            'Cyclone_Wind_Lag14d': recent['Route_Max_cyclone_wind'].iloc[-1],
            'Cyclone_Wind_Lag30d': recent['Route_Max_cyclone_wind'].iloc[-1],
            'Cyclone_Active_Lag7d': recent['Route_cyclone_active'].iloc[-1],
            'Cyclone_Active_Lag14d': recent['Route_cyclone_active'].iloc[-1],
            'Cyclone_Active_Lag30d': recent['Route_cyclone_active'].iloc[-1],
            'Cyclone_Wind_Forecast3d': recent['Route_Max_cyclone_wind'].mean(),
            'Cyclone_Wind_Forecast7d': recent['Route_Max_cyclone_wind'].mean(),
            'Cyclone_Active_Forecast3d': recent['Route_cyclone_active'].mean(),
        }
    
    def _get_weather_features(self, route, prediction_date):
        """Get weather features - these change daily but we use recent patterns"""
        route_data = self.historical_data[
            self.historical_data[f'Route_{route}'] == 1
        ].sort_values('Date')
        
        if len(route_data) == 0:
            return {}
        
        # Use recent weather patterns (last 7 days average)
        recent = route_data.tail(7)
        
        weather_features = {}
        weather_cols = [col for col in self.feature_names if any(x in col for x in 
            ['air_', 'slp_', 'wind_speed_', 'cyclone_wind_', 'cyclone_slp_', 'cyclone_dist_', 'cyclone_active_'])
            and not col.startswith('Route_') and not col.startswith('Cyclone_') and not col.startswith('Wind_')]
        
        for col in weather_cols:
            if col in recent.columns:
                weather_features[col] = recent[col].mean()
        
        # Wind speed forecast
        wind_cols = [col for col in self.feature_names if 'wind_speed_' in col and not col.startswith('Route_')]
        avg_wind = np.mean([recent[col].mean() for col in wind_cols if col in recent.columns])
        weather_features['Wind_Speed_Forecast7d'] = avg_wind
        weather_features['Wind_Speed_Lag7d'] = avg_wind
        weather_features['Wind_Speed_Lag14d'] = avg_wind
        weather_features['Wind_Speed_Lag30d'] = avg_wind
        
        return weather_features
    
    def _get_static_features(self, route):
        """Get features that don't change often (LSCI, SWI, etc.)"""
        route_data = self.historical_data[
            self.historical_data[f'Route_{route}'] == 1
        ]
        
        if len(route_data) == 0:
            return {}
        
        latest = route_data.iloc[-1]
        
        return {
            'Origin_LSCI': latest.get('Origin_LSCI', 55.2),
            'Dest_LSCI': latest.get('Dest_LSCI', 55.2),
            'Route_Mean_LSCI': latest.get('Route_Mean_LSCI', 55.2),
            'SWI': latest.get('SWI', 0.000025),
            'Coastal_Threat': latest.get('Coastal_Threat', 0),
            'Congestion_Risk': latest.get('Congestion_Risk', 0.3),
            'LSCI_30d_Delta': latest.get('LSCI_30d_Delta', 0),
        }
    
    def predict_single(self, route, prediction_date):
        """Generate prediction for a single route and date"""
        template = self._get_template_row(route, prediction_date)
        if template is None:
            return None
        
        # Combine all features
        features = {}
        features.update(self._get_static_features(route))
        features.update(self._get_weather_features(route, prediction_date))
        features.update(self._get_cyclone_features(route, prediction_date))
        features.update(self._calculate_lag_features(route, prediction_date))
        
        # Add route indicator
        for r in self.routes:
            features[f'Route_{r}'] = 1 if r == route else 0
        
        # Fill any missing features with defaults
        for fname in self.feature_names:
            if fname not in features:
                if fname in template.columns:
                    features[fname] = template[fname].values[0]
                else:
                    features[fname] = 0.0
        
        # Create feature vector in correct order
        feature_vector = [features.get(f, 0.0) for f in self.feature_names]
        
        # Predict
        import xgboost as xgb
        dmatrix = xgb.DMatrix([feature_vector], feature_names=self.feature_names)
        prediction = float(self.model.predict(dmatrix)[0])
        
        return {
            'route': route,
            'date': prediction_date.isoformat(),
            'predicted_price': round(prediction, 2),
            'confidence': self._estimate_confidence(prediction_date)
        }
    
    def _estimate_confidence(self, prediction_date):
        """Estimate prediction confidence based on days from training data"""
        pred_ts = pd.Timestamp(prediction_date) if not isinstance(prediction_date, pd.Timestamp) else prediction_date
        days_from_training = (pred_ts - self.last_training_date).days
        
        if days_from_training == 0:
            return 0.95
        elif days_from_training <= 7:
            return 0.90
        elif days_from_training <= 14:
            return 0.80
        elif days_from_training <= 30:
            return 0.70
        else:
            return 0.60
    
    def predict_range(self, start_date, end_date, routes=None):
        """Generate predictions for a date range"""
        if routes is None:
            routes = self.routes
        
        predictions = []
        current_date = pd.Timestamp(start_date) if not isinstance(start_date, pd.Timestamp) else start_date
        end_ts = pd.Timestamp(end_date) if not isinstance(end_date, pd.Timestamp) else end_date
        
        while current_date <= end_ts:
            # Skip weekends (shipping rates typically don't update)
            if current_date.weekday() < 5:  # Monday-Friday
                for route in routes:
                    pred = self.predict_single(route, current_date)
                    if pred:
                        predictions.append(pred)
            
            current_date += timedelta(days=1)
        
        return predictions
    
    def predict_today(self, routes=None):
        """Generate predictions for today"""
        today = pd.Timestamp(datetime.now().date())
        return self.predict_range(today, today, routes)
    
    def get_historical_summary(self):
        """Get summary of historical data"""
        summary = {}
        for route in self.routes:
            route_data = self.historical_data[
                self.historical_data[f'Route_{route}'] == 1
            ]
            if len(route_data) > 0:
                prices = route_data['Price_USD']
                summary[route] = {
                    'mean': float(round(prices.mean(), 2)),
                    'median': float(round(prices.median(), 2)),
                    'min': float(round(prices.min(), 2)),
                    'max': float(round(prices.max(), 2)),
                    'std': float(round(prices.std(), 2)),
                    'count': int(len(prices))
                }
        return summary
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_type': 'XGBoost',
            'features': len(self.feature_names),
            'last_training_date': self.last_training_date.isoformat(),
            'routes': self.routes,
            'test_metrics': {
                'MAE': 368.21,
                'RMSE': 499.93,
                'R2': 0.942,
                'MAPE': 8.50
            }
        }


# Singleton instance
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = ShippingPredictor()
    return _predictor


if __name__ == "__main__":
    predictor = get_predictor()
    
    print("\n=== Model Info ===")
    print(json.dumps(predictor.get_model_info(), indent=2))
    
    print("\n=== Historical Summary ===")
    summary = predictor.get_historical_summary()
    for route, stats in summary.items():
        print(f"\n{route}:")
        print(f"  Mean: ${stats['mean']:.0f}, Median: ${stats['median']:.0f}")
        print(f"  Range: ${stats['min']:.0f} - ${stats['max']:.0f}")
    
    print("\n=== Today's Predictions ===")
    today_preds = predictor.predict_today()
    for pred in today_preds:
        print(f"  {pred['route']}: ${pred['predicted_price']:.0f} (confidence: {pred['confidence']:.0%})")
