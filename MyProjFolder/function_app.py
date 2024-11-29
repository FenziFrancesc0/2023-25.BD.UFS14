import azure.functions as func
import json
import logging

app = func.FunctionApp()

@app.route(route="SimulaPrestito", auth_level=func.AuthLevel.ANONYMOUS)
def SimulaPrestito(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Simulatore di Prestito')
    
    try:
        importo = float(req.params.get('importo'))
        tasso = float(req.params.get('tasso')) / 100
        anni = int(req.params.get('anni'))
        
        rate_mensili = anni * 12
        rata = (importo * tasso / 12) / (1 - (1 + tasso / 12) ** -rate_mensili)
        
        return func.HttpResponse(json.dumps({
            "importo": importo,
            "tasso": tasso * 100,
            "anni": anni,
            "rata_mensile": round(rata, 2)
        }), status_code=200, mimetype="application/json")
    except Exception as e:
        logging.error(f"Errore: {e}")
        return func.HttpResponse("Richiesta non valida. Controlla i dati inseriti.", status_code=400)