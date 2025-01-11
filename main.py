import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from flask import Flask, request, jsonify, render_template
from web3 import Web3
from dotenv import load_dotenv
import os

load_dotenv()

infura_url = 'https://sepolia.infura.io/v3/3c5b238bb8ae42e9a76ddd561cd51de8'
web3 = Web3(Web3.HTTPProvider(infura_url))

if web3.is_connected():
    print("Успешное подключение к сети Sepolia")
else:
    print("Не удалось подключиться")

contract_address = '0xEf9f1ACE83dfbB8f559Da621f4aEA72C6EB10eBf'
contract_abi = [
    {
        "inputs": [
            {"internalType": "address", "name": "user", "type": "address"},
            {"internalType": "uint256", "name": "risk", "type": "uint256"},
            {"internalType": "uint256", "name": "probability", "type": "uint256"}
        ],
        "name": "storePrediction",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "user", "type": "address"}
        ],
        "name": "getPrediction",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"},
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

contract = web3.eth.contract(address=contract_address, abi=contract_abi)

data = pd.read_csv("cs-training.csv", index_col=0)

data = data.dropna()
features = data.drop(["SeriousDlqin2yrs"], axis=1)
target = data["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, probabilities))
print("Classification Report:\n", classification_report(y_test, predictions))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_risk():
    try:
        data = request.form
        df = pd.DataFrame({
            "RevolvingUtilizationOfUnsecuredLines": [float(data["RevolvingUtilizationOfUnsecuredLines"])],
            "age": [int(data["age"])],
            "NumberOfTime30-59DaysPastDueNotWorse": [int(data["NumberOfTime30-59DaysPastDueNotWorse"])],
            "DebtRatio": [float(data["DebtRatio"])],
            "MonthlyIncome": [float(data["MonthlyIncome"])],
            "NumberOfOpenCreditLinesAndLoans": [int(data["NumberOfOpenCreditLinesAndLoans"])],
            "NumberOfTimes90DaysLate": [int(data["NumberOfTimes90DaysLate"])],
            "NumberRealEstateLoansOrLines": [int(data["NumberRealEstateLoansOrLines"])],
            "NumberOfTime60-89DaysPastDueNotWorse": [int(data["NumberOfTime60-89DaysPastDueNotWorse"])],
            "NumberOfDependents": [int(data["NumberOfDependents"])]
        })

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0, 1]

        web3.eth.default_account = "0x34F7E8AEFDBb9D77532B9642ee2687E2d85598F0"
        private_key = os.getenv("PRIVATE_KEY")
        probability_uint256 = int(probability * 100)

        tx = contract.functions.storePrediction(
            web3.eth.default_account,
            int(prediction),
            probability_uint256
        ).build_transaction({
            'from': web3.eth.default_account,
            'gas': 200000,
            'gasPrice': web3.to_wei('20', 'gwei'),
            'nonce': web3.eth.get_transaction_count(web3.eth.default_account),
        })

        signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)

        return render_template('result.html', risk=int(prediction), probability=probability, transaction_hash=tx_hash.hex())
    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
