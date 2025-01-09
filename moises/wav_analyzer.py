import Spectrum

a = Spectrum.Spectrum("parte1/pulita_difficile.wav")

a.plot()
a.drop_notes()
a.mean_synthesis("sintetizzati/pulita_difficile_s.wav", first=True)
a.peak_synthesis(2, "sintetizzati/pulita_difficile_m.wav", first=True)