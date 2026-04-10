from pathlib import Path

import pandas as pd


RAW_DIR = Path(__file__).resolve().parent / "raw"
OUTPUT_PATH = Path(__file__).resolve().parent / "hospital_data.csv"
SAMPLE_SIZE = 300
RANDOM_STATE = 42


def _load_general_info() -> pd.DataFrame:
	path = RAW_DIR / "hospital_general_information.csv"
	df = pd.read_csv(path)
	cols = [
		"facility_id",
		"facility_name",
		"hospital_overall_rating",
		"count_of_safety_measures_better",
		"count_of_readm_measures_worse",
	]
	return df[cols]


def _load_measure(path: Path, measure_id: str, output_col: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	df = df.loc[df["measure_id"] == measure_id, ["facility_id", "score"]].copy()
	df = df.rename(columns={"score": output_col})
	return df.drop_duplicates(subset=["facility_id"])


def build_hospital_data(sample_size: int = SAMPLE_SIZE, random_state: int = RANDOM_STATE) -> pd.DataFrame:
	a = _load_general_info()

	b_path = RAW_DIR / "complications_and_deaths_hospital.csv"
	d_path = RAW_DIR / "medicare_hospital_spending_per_patient_hospital.csv"
	e_path = RAW_DIR / "unplanned_hospital_visits_hospital.csv"

	mort_ami = _load_measure(b_path, measure_id="MORT_30_AMI", output_col="mort_ami")
	comp_hip_knee = _load_measure(
		b_path,
		measure_id="COMP_HIP_KNEE",
		output_col="comp_hip_knee",
	)
	spending = _load_measure(d_path, measure_id="MSPB-1", output_col="spending")
	readmission_hf = _load_measure(e_path, measure_id="EDAC_30_HF", output_col="readmission_hf")

	merged = (
		a.merge(mort_ami, on="facility_id", how="inner")
		.merge(comp_hip_knee, on="facility_id", how="inner")
		.merge(spending, on="facility_id", how="inner")
		.merge(readmission_hf, on="facility_id", how="inner")
	)

	numeric_cols = [
		"hospital_overall_rating",
		"count_of_safety_measures_better",
		"count_of_readm_measures_worse",
		"mort_ami",
		"comp_hip_knee",
		"spending",
		"readmission_hf",
	]
	for col in numeric_cols:
		merged[col] = pd.to_numeric(merged[col], errors="coerce")

	merged = merged.dropna()

	final_cols = [
		"facility_id",
		"facility_name",
		"mort_ami",
		"comp_hip_knee",
		"readmission_hf",
		"spending",
		"count_of_safety_measures_better",
		"count_of_readm_measures_worse",
		"hospital_overall_rating",
	]
	result = merged[final_cols].copy()
	result["facility_id"] = result["facility_id"].astype("int64")

	if len(result) > sample_size:
		result = result.sample(n=sample_size, random_state=random_state)

	return result.sort_values("facility_id").reset_index(drop=True)


def main() -> None:
	hospital_data = build_hospital_data()
	hospital_data.to_csv(OUTPUT_PATH, index=False)
	print(f"Saved {len(hospital_data)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
