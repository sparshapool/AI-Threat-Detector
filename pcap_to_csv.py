import pyshark
import pandas as pd

# Corrected path to the pcapng file
pcap_file = "C:/Users/Sparsh/Documents/AI-Threat-Detector/Data.pcapng"

# Read the pcapng file using pyshark (filtering for IP packets)
cap = pyshark.FileCapture(pcap_file, display_filter="ip")

# Extract relevant information from each packet
data = []
for packet in cap:
    try:
        data.append({
            "No.": packet.number,
            "Time": packet.sniff_time,
            "Source": packet.ip.src,
            "Destination": packet.ip.dst,
            "Protocol": packet.highest_layer,
            "Length": packet.length
        })
    except AttributeError:
        # Skip packets that do not have the required attributes
        continue

# Convert the list of dictionaries into a pandas DataFrame
df = pd.DataFrame(data)

# Define the output CSV file path (kept the same)
csv_file = "C:/Users/Sparsh/Documents/AI-Threat-Detector/network_data.csv"

# Save the DataFrame to a CSV file
df.to_csv(csv_file, index=False)

print("✅ Conversion complete! Data saved to:", csv_file)
