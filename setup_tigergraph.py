import pandas as pd
import numpy as np
import sys
import requests
import json
import time

BASE_URL = "http://localhost:9000"
GSQL_URL = "http://localhost:14240"
AUTH = ("tigergraph", "tigergraph")


def restpp_get(path, params=None):
    r = requests.get(f"{BASE_URL}{path}", auth=AUTH, params=params, timeout=30)
    return r


def restpp_post(path, data=None, json_data=None):
    r = requests.post(f"{BASE_URL}{path}", auth=AUTH, data=data, json=json_data, timeout=60)
    return r


def gsql_post(command):
    """Submit GSQL via the internal GSQL server on port 14240."""
    try:
        r = requests.post(
            f"{GSQL_URL}/gsql/v1/statements",
            auth=AUTH,
            data=command.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            timeout=120
        )
        return r.status_code, r.text
    except Exception as e:
        return None, str(e)


def wait_for_server(max_retries=20, delay=10):
    print("Waiting for TigerGraph REST++ to be ready...")
    for i in range(max_retries):
        try:
            r = requests.get(f"{BASE_URL}/echo", auth=AUTH, timeout=5)
            if r.status_code == 200:
                print(f"  REST++ is ready! ({r.json().get('message', 'OK')})")
                return True
        except Exception:
            pass
        print(f"  Attempt {i+1}/{max_retries} - not ready yet, waiting {delay}s...")
        time.sleep(delay)
    return False


def check_gsql_available():
    """Check if GSQL port 14240 is reachable."""
    try:
        r = requests.get(f"{GSQL_URL}/gsql/v1/version", auth=AUTH, timeout=5)
        return r.status_code in [200, 401]
    except Exception:
        return False


def create_schema_via_gsql():
    """Submit schema GSQL via port 14240."""
    print("\nCreating schema via GSQL (port 14240)...")
    schema_gsql = """
CREATE VERTEX Customer (PRIMARY_ID customer_id STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX Card (PRIMARY_ID card_id STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX Transaction (PRIMARY_ID TransactionID STRING, ts DATETIME, TransactionAmt DOUBLE, ProductCD STRING, channel STRING, risk_score DOUBLE) WITH primary_id_as_attribute="true"
CREATE VERTEX DeviceProfile (PRIMARY_ID device_id STRING, DeviceInfo STRING, id_30 STRING, id_31 STRING, id_33 STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX EmailDomain (PRIMARY_ID email_domain STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX BillingRegion (PRIMARY_ID region_code STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX ClosedCase (PRIMARY_ID case_id STRING, outcome STRING, pattern STRING, exposure_usd DOUBLE) WITH primary_id_as_attribute="true"
CREATE DIRECTED EDGE OWNS (FROM Customer, TO Card)
CREATE DIRECTED EDGE MADE (FROM Card, TO Transaction)
CREATE DIRECTED EDGE FROM_DEVICE (FROM Transaction, TO DeviceProfile)
CREATE DIRECTED EDGE PURCHASER_EMAIL (FROM Transaction, TO EmailDomain)
CREATE DIRECTED EDGE BILLED_IN (FROM Transaction, TO BillingRegion)
CREATE DIRECTED EDGE NEXT (FROM Transaction, TO Transaction)
CREATE DIRECTED EDGE INVOLVES (FROM ClosedCase, TO Transaction)
CREATE DIRECTED EDGE ON_CARD (FROM ClosedCase, TO Card)
CREATE DIRECTED EDGE CONNECTED_TO (FROM ClosedCase, TO Card)
CREATE GRAPH FraudGraph (Customer, Card, Transaction, DeviceProfile, EmailDomain, BillingRegion, ClosedCase, OWNS, MADE, FROM_DEVICE, PURCHASER_EMAIL, BILLED_IN, NEXT, INVOLVES, ON_CARD, CONNECTED_TO)
"""
    status, result = gsql_post(schema_gsql)
    print(f"  GSQL response ({status}): {str(result)[:500]}")
    return status == 200 and "error" not in str(result).lower()


def process_and_ingest_transactions(graph, file_path):
    print(f"\n--- Ingesting Transactions from {file_path} ---")
    chunk_size = 10000
    total_rows = 0

    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        chunk.fillna("", inplace=True)
        chunk["card_id"] = chunk["card1"].astype(str) + "_" + chunk["card2"].astype(str)

        for col in ["DeviceInfo", "id_30", "id_31", "id_33"]:
            if col not in chunk.columns:
                chunk[col] = ""
        chunk["device_id"] = (
            chunk["DeviceInfo"].astype(str) + "|" +
            chunk["id_30"].astype(str) + "|" +
            chunk["id_31"].astype(str) + "|" +
            chunk["id_33"].astype(str)
        )

        def batch_upsert_vertices(vtype, records):
            if not records:
                return
            vertices_dict = {}
            for rec in records:
                vid = str(rec.pop("_id"))
                vertices_dict[vid] = {"attributes": {k: v for k, v in rec.items()}}
            restpp_post(f"/graph/{graph}", json_data={"vertices": {vtype: vertices_dict}})

        def batch_upsert_edges(src_type, etype, tgt_type, pairs):
            if not pairs:
                return
            edges_dict = {src_type: {}}
            for src, tgt in pairs:
                src, tgt = str(src), str(tgt)
                if src not in edges_dict[src_type]:
                    edges_dict[src_type][src] = {etype: {tgt_type: {}}}
                edges_dict[src_type][src][etype][tgt_type][tgt] = {}
            restpp_post(f"/graph/{graph}", json_data={"edges": edges_dict})

        if "customer_id" in chunk.columns:
            batch_upsert_vertices("Customer", [{"_id": r, "customer_id": r} for r in chunk["customer_id"].unique() if r != ""])
        batch_upsert_vertices("Card", [{"_id": r, "card_id": r} for r in chunk["card_id"].unique()])

        tx_dict = {}
        for _, row in chunk.iterrows():
            tx_dict[str(row["TransactionID"])] = {"attributes": {
                "TransactionAmt": float(row["TransactionAmt"]) if row["TransactionAmt"] != "" else 0.0,
                "ProductCD": str(row.get("ProductCD", "")),
                "channel": str(row.get("channel", "")),
                "risk_score": float(row.get("risk_score", 0.0)) if row.get("risk_score", "") != "" else 0.0,
            }}
        restpp_post(f"/graph/{graph}", json_data={"vertices": {"Transaction": tx_dict}})

        dev_dict = {}
        for _, row in chunk.drop_duplicates("device_id").iterrows():
            if row["device_id"]:
                dev_dict[str(row["device_id"])] = {"attributes": {
                    "DeviceInfo": str(row.get("DeviceInfo", "")),
                    "id_30": str(row.get("id_30", "")),
                    "id_31": str(row.get("id_31", "")),
                    "id_33": str(row.get("id_33", "")),
                }}
        if dev_dict:
            restpp_post(f"/graph/{graph}", json_data={"vertices": {"DeviceProfile": dev_dict}})

        if "P_emaildomain" in chunk.columns:
            emails = [e for e in chunk["P_emaildomain"].unique() if e != ""]
            if emails:
                restpp_post(f"/graph/{graph}", json_data={"vertices": {"EmailDomain": {e: {"attributes": {"email_domain": e}} for e in emails}}})

        if "addr1" in chunk.columns:
            regions = [str(r) for r in chunk["addr1"].unique() if r != ""]
            if regions:
                restpp_post(f"/graph/{graph}", json_data={"vertices": {"BillingRegion": {r: {"attributes": {"region_code": r}} for r in regions}}})

        if "customer_id" in chunk.columns:
            batch_upsert_edges("Customer", "OWNS", "Card", list(zip(chunk["customer_id"], chunk["card_id"])))
        batch_upsert_edges("Card", "MADE", "Transaction", list(zip(chunk["card_id"], chunk["TransactionID"].astype(str))))
        batch_upsert_edges("Transaction", "FROM_DEVICE", "DeviceProfile", list(zip(chunk["TransactionID"].astype(str), chunk["device_id"])))

        if "P_emaildomain" in chunk.columns:
            email_pairs = [(tx, em) for tx, em in zip(chunk["TransactionID"].astype(str), chunk["P_emaildomain"]) if em != ""]
            batch_upsert_edges("Transaction", "PURCHASER_EMAIL", "EmailDomain", email_pairs)

        if "addr1" in chunk.columns:
            region_pairs = [(tx, str(r)) for tx, r in zip(chunk["TransactionID"].astype(str), chunk["addr1"]) if str(r) != ""]
            batch_upsert_edges("Transaction", "BILLED_IN", "BillingRegion", region_pairs)

        total_rows += len(chunk)
        print(f"  Ingested {total_rows} rows...")

    print(f"Transaction ingestion complete. Total: {total_rows} rows.")


def process_and_ingest_closed_cases(graph, file_path):
    print(f"\n--- Ingesting Closed Cases from {file_path} ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"  File not found: {file_path}, skipping.")
        return

    df.fillna("", inplace=True)
    df["exposure_usd"] = pd.to_numeric(df.get("exposure_usd", 0), errors="coerce").fillna(0.0)

    case_dict = {}
    for _, row in df.iterrows():
        case_dict[str(row["case_id"])] = {"attributes": {
            "outcome": str(row.get("outcome", "")),
            "pattern": str(row.get("pattern", "")),
            "exposure_usd": float(row.get("exposure_usd", 0.0)),
        }}
    restpp_post(f"/graph/{graph}", json_data={"vertices": {"ClosedCase": case_dict}})

    if "TransactionID" in df.columns:
        edges_dict = {"ClosedCase": {}}
        for _, row in df.iterrows():
            if row["TransactionID"] != "":
                src, tgt = str(row["case_id"]), str(row["TransactionID"])
                if src not in edges_dict["ClosedCase"]:
                    edges_dict["ClosedCase"][src] = {"INVOLVES": {"Transaction": {}}}
                edges_dict["ClosedCase"][src]["INVOLVES"]["Transaction"][tgt] = {}
        restpp_post(f"/graph/{graph}", json_data={"edges": edges_dict})

    print(f"  Ingested {len(df)} closed cases.")


if __name__ == "__main__":
    # Step 1: Wait for server
    if not wait_for_server():
        print("ERROR: TigerGraph REST++ never became ready. Exiting.")
        sys.exit(1)

    # Step 2: Check if GSQL port 14240 is available
    gsql_available = check_gsql_available()
    print(f"\nGSQL port 14240 available: {gsql_available}")

    graph = "FraudGraph"

    if gsql_available:
        success = create_schema_via_gsql()
        if not success:
            print("GSQL schema failed. You may need to expose port 14240 via Docker.")
    else:
        print("\nGSQL port 14240 not exposed externally.")
        print("To create the schema, restart the container with port 14240 mapped:")
        print("  sudo docker rm -f tigergraph")
        print("  sudo docker run -d -p 9000:9000 -p 14240:14240 --name tigergraph \\")
        print("    --ulimit nofile=1000000:1000000 -m 8g --shm-size 2g tigergraph/tigergraph:3.10.4")
        print("\nThen re-run this script. If the graph already exists, ingestion will proceed.")

    # Step 3: Check if the graph is accessible
    print(f"\nChecking graph '{graph}'...")
    r = restpp_get(f"/graph/{graph}/vertices/Transaction", {"limit": 1})
    if r.status_code == 200:
        print(f"  Graph '{graph}' is ready! Starting data ingestion...")
        process_and_ingest_transactions(graph, "Data Set/cleaned_transactions_merged.csv")
        process_and_ingest_closed_cases(graph, "Data Set/closed_cases_history.csv")
        print("\n✅ Setup and ingestion complete!")
    else:
        print(f"  Graph not yet accessible: {r.status_code} - {r.text[:200]}")
        print("\nAction required: Recreate the container with port 14240 exposed (see above).")
