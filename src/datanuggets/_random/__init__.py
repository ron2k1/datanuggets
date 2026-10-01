"""Sources of randomness. Users get numpy by default (numpy_draws), RDraws draws exactly what R
draws (r_draws), and parity tests replay the draws R recorded (replay). Each one implements the
Draws interface (interfaces), and seeds turns a seed= argument into one of them."""
