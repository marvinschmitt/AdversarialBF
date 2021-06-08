############################
## Author: Marvin Schmitt ##
############################


# Load Libaries
library(tidyverse)

# Set Working directory
setwd(dirname(rstudioapi::getSourceEditorContext()$path))


N = 100000
x = rnorm(N)


epsilon = rnorm(N)
alpha = 0.01
xi = epsilon[pnorm(-abs(epsilon)) < alpha]
col = c(rep("green", 54), 
        rep("grey", 92),
        rep("green", 54))

hist(epsilon, breaks=seq(-5,5,l=200), col=col, freq=FALSE)

hist(c(x, xi), breaks=seq(-5,5,l=200),  col=col, freq=FALSE)
