from gwpy.timeseries import TimeSeries
ip_file="h1_noise_raw.hdf5"
op_file="h1_noise_processed.hdf5"
noise_data=TimeSeries.read(ip_file)
#whitening: flatten the noise spectrum (based on PSD: how much power exists at each frequency)
whiten_length=4
overlap_length=2
noise_data = noise_data.whiten(fftlength=whiten_length, overlap=overlap_length)
#fftlength: length of the segment used to estimate the PSD
#overlap: overlap between segments used to estimate the PSD
#bandpass filter:range the frequency of interest (20-500Hz), 20: environmental noise, 500: detector sensitivity
noise_data = noise_data.bandpass(20, 500)
#resample: reduce the sample rate to 1024Hz, to reduce the data size and speed up processing
noise_data = noise_data.resample(1024)
#save the processed data to a new file
noise_data.write(op_file, overwrite=True)
print(f"\nSaved processed noise segment:")
print(f"  GPS range: {noise_data.t0.value} - {noise_data.t0.value+noise_data.duration.value}")
print(f"  Duration: {noise_data.duration}")
print(f"  Sample rate: {noise_data.sample_rate}")       