from FrequencyEncoders import FrequencyEncoder
from FFTEncoders import FFTEncoder
from KMersEncoders import KMersEncoders
from OneHotEncoder import OneHotEncoder
from OrdinalEncoder import OrdinalEncoder
from PhysicochemicalEncoder import PhysicochemicalEncoder
import pandas as pd
import sys

print("Reading data")
df_data = pd.read_csv(sys.argv[1])
path_export = sys.argv[2]
name_response = sys.argv[3]
df_properties = pd.read_csv(sys.argv[4])
max_length = int(sys.argv[5])

df_properties.index = df_properties["residue"].values

print("Encoding by frequencies")
frequency_instance = FrequencyEncoder(
    dataset=df_data, 
    sequence_column="sequence", 
    ignore_columns=[name_response],
    max_length=max_length
)

frequency_instance.run_process()
frequency_instance.coded_dataset.to_csv(f"{path_export}frequency_encoder.csv", index=False)

print("Encoding by One Hot")
one_hot_instance = OneHotEncoder(
    dataset=df_data, 
    sequence_column="sequence", 
    ignore_columns=[name_response],
    max_length=max_length
)

one_hot_instance.run_process()
one_hot_instance.coded_dataset.to_csv(f"{path_export}one_hot_encoder.csv", index=False)

print("Encoding by Ordinal Encoder")
ordinal_instance = OrdinalEncoder(
    dataset=df_data, 
    sequence_column="sequence", 
    ignore_columns=[name_response],
    max_length=max_length
)

ordinal_instance.run_process()
ordinal_instance.coded_dataset.to_csv(f"{path_export}ordinal_encoder.csv", index=False)

print ("Processing physicochemical properties and FFT")
for i in range(8):
    name_property = f"Group_{i}"

    physicochemical_instance = PhysicochemicalEncoder(
        dataset=df_data, 
        sequence_column="sequence", 
        ignore_columns=[name_response],
        max_length=max_length,
        name_property=name_property,
        df_properties=df_properties
    )

    physicochemical_instance.run_process()

    physicochemical_instance.df_data_encoded.to_csv(f"{path_export}physicochemical_property_{i}.csv", index=False)

    fft_instance = FFTEncoder(
        dataset=physicochemical_instance.df_data_encoded, 
        sequence_column="sequence", 
        ignore_columns=[name_response]
    )

    fft_instance.encoding_dataset()
    fft_instance.df_fft.to_csv(f"{path_export}FFT_PHY_{i}.csv", index=False)

print("Apply K-mers")

for j in range(3, 6):
    print("Processing K-mers: ", j)
    kmer_instance = KMersEncoders(
        dataset=df_data, 
        sequence_column="sequence", 
        ignore_columns=[name_response],
        size_kmer=j
    )

    kmer_instance.process_dataset()
    kmer_instance.coded_dataset.to_csv(f"{path_export}kmers_{j}.csv", index=False)