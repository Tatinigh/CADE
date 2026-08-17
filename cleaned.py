from gwpy.timeseries import TimeSeries
import requests #api call once
 #config from event list 
detector='H1'
O3A_START=1238166018 
O3A_END=1253977218
WINDOW_START = 1240303737.2
WINDOW_END = WINDOW_START + 32
window_duration=32
buffer=10
SAMPLE_RATE=4096
#no of event gps in 03a window to avoid overlap with noise
response = requests.get("https://gwosc.org/eventapi/json/GWTC/", timeout=30)
response.raise_for_status()
event = response.json()
event_times=[]
for name,info in event['events'].items():
    gps = info.get('GPS')
    if gps and O3A_START<= gps <= O3A_END:
        event_times.append(gps)
print(f'Found {len(event_times)} events in O3a window')

#is chosen window is clean?
def is_clean(WINDOW_START, WINDOW_END, event_times, buffer):
    for T in event_times:
        if (WINDOW_START - buffer) <= T <= (WINDOW_END + buffer):
            return False,T
    return True,None
clean,conflict_time = is_clean(WINDOW_START, WINDOW_END, event_times, buffer) #store kore boolean value and time of conflict jodi thake
if clean:
    print(f'Window {WINDOW_START} to {WINDOW_END} is clean')
else:
    print(f'Window {WINDOW_START} to {WINDOW_END} is not clean, event at {conflict_time}')

#clean data fetchingggg
noise=TimeSeries.fetch_open_data(detector, WINDOW_START, WINDOW_END, sample_rate=SAMPLE_RATE)
noise.write('h1_noise_raw.hdf5',overwrite=True)

print(f"\nSaved noise segment:")
print(f"  GPS range: {WINDOW_START} - {WINDOW_END}")
print(f"  Duration: {noise.duration}")
print(f"  Sample rate: {noise.sample_rate}")
print(f"  Data points: {len(noise)}")
