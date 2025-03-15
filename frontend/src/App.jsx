import { useState } from "react";

function App() {
  const [prediction, setPrediction] = useState("");

  const handlePredict = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {  // ✅ Use correct backend URL
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          "Protocol_TCP": 1,
          "Protocol_TLS": 0,
          "Total_Source_Freq": 150,
          "Total_Destination_Freq": 20
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch prediction");
      }

      const data = await response.json();
      setPrediction(data.prediction || "Error in prediction");
    } catch (error) {
      console.error("Error fetching prediction:", error);
      setPrediction("Error connecting to backend");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>AI Threat Detector</h1>
      <button onClick={handlePredict}>Predict Threat</button>
      <p><strong>Prediction:</strong> {prediction}</p>
    </div>
  );
}

export default App;
