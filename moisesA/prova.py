import SpectrumA as Sp

file = 3
a = Sp.Spectrum(f"data{file}.txt")

a.plot(f"data{file}", c="limegreen", zoom=2)
a.DIY_synthesis(0, f"data{file}", c="limegreen", zoom=2)