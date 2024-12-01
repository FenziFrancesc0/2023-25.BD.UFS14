import azure.functions as func
import logging

app = func.FunctionApp()

@app.route(route="SimulaPrestito", auth_level=func.AuthLevel.ANONYMOUS)
def SimulaPrestito(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Simulatore di Prestito')

    try:
        importo = float(req.params.get('importo', 10000))
        tasso = float(req.params.get('tasso', 5)) / 100
        anni = int(req.params.get('anni', 5))
        
        rate_mensili = anni * 12
        if tasso > 0:
            rata = (importo * tasso / 12) / (1 - (1 + tasso / 12) ** -rate_mensili)
        else:
            rata = importo / rate_mensili

        html_content = f"""
        <!DOCTYPE html>
        <html lang="it">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Simulatore di Prestito</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .container {{ max-width: 600px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                input {{ width: 100%; padding: 8px; margin: 5px 0; }}
                button {{ background-color: #0078D4; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer; }}
                button:hover {{ background-color: #005A9E; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Risultato Simulazione Prestito</h1>
                <p><strong>Importo:</strong> €{importo:.2f}</p>
                <p><strong>Tasso:</strong> {tasso * 100:.2f}%</p>
                <p><strong>Anni:</strong> {anni} anni</p>
                <p><strong>Rata Mensile:</strong> €{rata:.2f}</p>
                <hr>
                <h2>Prova altri valori</h2>
                <form method="GET" action="/api/SimulaPrestito">
                    <label for="importo">Importo:</label>
                    <input type="number" id="importo" name="importo" value="{importo}" required>
                    <label for="tasso">Tasso (%)</label>
                    <input type="number" id="tasso" name="tasso" value="{tasso * 100}" step="0.01" required>
                    <label for="anni">Anni</label>
                    <input type="number" id="anni" name="anni" value="{anni}" required>
                    <button type="submit">Calcola</button>
                </form>
            </div>
        </body>
        </html>
        """

        return func.HttpResponse(html_content, status_code=200, mimetype="text/html")
    except ValueError as e:
        logging.error(f"Errore di validazione dei dati: {e}")
        return func.HttpResponse(
            "I dati forniti non sono validi. Assicurati che 'importo', 'tasso' e 'anni' siano numerici.",
            status_code=400
        )
    except Exception as e:
        logging.error(f"Errore generale: {e}")
        return func.HttpResponse(
            "Errore, riprovare più tardi.",
            status_code=500
        )
