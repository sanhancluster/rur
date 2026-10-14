# Snapshot parameters
# REFOUT = 828 # Reference iout
REFOUT = None # if None, auto-detect the latest output

# Paramteres for prunning halo catalogs
FCONTAM = 0.001 # Mcontam/Mhalo < FCONTAM
HLVL = 2 # Halo level <= HLVL

# Parameters for Galaxy-Halo Matching
HMIN = 1e13 # Select Host halo with Mvir > HMIN
MATCHLVL = 1 # 0: Strict (gal near center), 1: gal inside hal, 2: Loose (gal-hal touched)
GLVL = 2 # Galaxy level <= GLVL

# Galaxy radius = GRADII*gal[GRADIUS]
GRADIUS = 'r90' # 'r', 'r50', 'r90'
GRADII = 2
