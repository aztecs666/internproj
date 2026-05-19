import urllib.request, json

def get(path):
    resp = urllib.request.urlopen('http://localhost:8000' + path)
    return json.loads(resp.read())

print('=== 1. Root ===')
print(get('/'))

print('\n=== 2. Model Info ===')
info = get('/api/model-info')
print('Model:', info['model_type'])
print('Features:', info['features'])
print('Routes:', info['routes'])

print('\n=== 3. Today Predictions ===')
today = get('/api/predictions/today')
for p in today['predictions']:
    print(f'  {p["route"]}: ${p["predicted_price"]:.0f} (confidence: {p["confidence"]:.0%})')

print('\n=== 4. Stream (last 7 days) ===')
data = get('/api/predictions/stream')
print('Start:', data['start_date'], 'to', data['end_date'])
print('Total predictions:', data['total_predictions'])
for route, preds in data['predictions_by_route'].items():
    print(f'  {route}: {len(preds)} predictions')

print('\n=== 5. Historical Summary ===')
summary = get('/api/historical-summary')
for route, stats in summary.items():
    print(f'  {route}: mean=${stats["mean"]:.0f}, max=${stats["max"]:.0f}')

print('\n=== 6. Routes ===')
routes = get('/api/routes')
for rid, name in routes['route_names'].items():
    print(f'  {rid}: {name}')

print('\n=== ALL TESTS PASSED ===')
