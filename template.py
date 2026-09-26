#pybc.waveform creates random gravitationa wave templates
from pycbc.waveform import get_td_waveform
#get_td_waveform is pybcs fxn for generating time-domain waveforms for gravitational waves. It allows users to specify parameters such as masses, spins, and other properties of the binary system to create a waveform that represents the gravitational wave signal produced by the merger of two compact objects, such as black holes or neutron stars.
import numpy as np
import pickle #for saving and loading Python objects to and from files
def generate_waveform(mass1, mass2, delta_t=1.0/1024, f_lower=20):
    #mass1, mass2 — the masses of the two merging black holes (in solar masses), no default value — you must supply these each time you call the function
    #delta_t=1.0/1024 — a default value if you don't specify one; equals 0.0009765625 seconds, which is the time gap between consecutive samples. This is just 1 / sample_rate, so 1/1024 means 1024 samples per second (1024 Hz)
    #f_lower=20 — another default; the starting frequency (in Hz) where the waveform calculation begins
    hp, hc= get_td_waveform(
        approximant='IMRPhenomD',  #the waveform model used for generating the gravitational wave signal. IMRPhenomD is a phenomenological model that describes the inspiral, merger, and ringdown phases of binary black hole coalescence.
        mass1=mass1,  #the mass of the first black hole in solar masses
        mass2=mass2,  #the mass of the second black hole in solar masses    
        delta_t=delta_t,  #the time interval between consecutive samples in seconds
        f_lower=f_lower   #the starting frequency (in Hz) where the waveform calculation begins
    )
    return hp #hp and hc together capture the full gravitational wave signal, but for your denoising project, you might only need hp. hp represents the plus polarization of the gravitational wave, while hc represents the cross polarization. Depending on your specific application, you may choose to use one or both polarizations.
    # hc is cross polarization, which is another component of the gravitational wave signal. In many cases, researchers focus on the plus polarization (hp) for analysis, but both polarizations can provide valuable information about the source and characteristics of the gravitational wave event.
NUM_SAMPLES = 500
templates = []
for i  in range(NUM_SAMPLES):
    m1 = np.random.uniform(10, 50)  # Randomly select mass1 between 10 and 50 solar masses
    m2 = np.random.uniform(10, 50)  # Randomly select mass2 between 10 and 50 solar masses
    waveform = generate_waveform(m1, m2)
    templates.append({'mass1': m1, 'mass2': m2, 'waveform': np.array(waveform)})  # Store the masses and the generated waveform in a dictionary and append it to the templates list
with open('waveform_templates.pkl', 'wb') as f:  # Open a file in write-binary mode to save the templates
    pickle.dump(templates, f)  # Serialize the templates list and write it to the file