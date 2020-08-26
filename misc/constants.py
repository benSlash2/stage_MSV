seed = 1  # random seed of 1
day_len = 1440  # number of samples per day, 1440 = 1 sample every minute
cv = 5  # outer cross-validation factor 4
path = "."  # current path, change if the main function is run from outside the project
freq = 5
ph = 30

K_DIA = 0.0182
C_bio = 0.8
t_max = 60
k_s = 0.0115

ph_f = ph // freq
day_len_f = day_len // freq

