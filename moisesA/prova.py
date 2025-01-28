import SpectrumA as Sp

file = 2
a = Sp.Spectrum(f"data{file}.txt")

a.plot(f"data{file}", c="tomato", zoom=1)
a.DIY_synthesis(0, f"data{file}", c="tomato", zoom=1)