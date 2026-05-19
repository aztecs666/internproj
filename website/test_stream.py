import urllib.request, json

resp = urllib.request.urlopen('http://localhost:8000/api/predictions/stream')
data = json.loads(resp.read())
print('Training cutoff:', data['training_cutoff'])
print('Today:', data['today'])
print('Days gap:', data['days_gap'])
print('Total predictions:', data['total_predictions'])

for route, preds in data['predictions_by_route'].items():
    dates = [p['date'] for p in preds]
    first_pred = preds[0]
    last_pred = preds[-1]
    print(f'{route}: {len(preds)} predictions')
    print(f'  From: {dates[0]} To: {dates[-1]}')
    print(f'  First: ${first_pred["predicted_price"]:.0f}, Last: ${last_pred["predicted_price"]:.0f}')
