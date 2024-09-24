############################################
# Additional Model code for Homework 1 
# this will serve as the code for the SCAD and MC+ penalty models, and we will use 
# some type of common score to compare the models fairly 
############################################ 

# loading the necessary libraries 
library("grpreg")
library("lars")
library("ncvreg")

# loading the data 
X <- read.csv("/home/adbucks/Documents/sta_629/Homework_1/X_SRBCT.csv", header = FALSE)
y <- read.csv("/home/adbucks/Documents/sta_629/Homework_1/Y_SRBCT.csv", header = FALSE) 
head(X)
dim(X)
dim(y)

data("Prostate")
class(Prostate$X)
class(Prostate$y)

# comparing 
class(X) # need to convert to matrix array 

X <- as.matrix(X)
class(y) # need to convert to numeric
y <- y$V1


# now we can do the actual modeling with the data loaded and get to plotting
# and eventually scoring 
# using another package to avoid the grouped covariates problem 
# need to coerce y into a numeric 
# trying another way 

# trying the new way 
mcp <- ncvreg(X, y) # MCP is default
coef(mcp, lambda = 0.05)
summary(mcp , lambda = 0.1)

# now trying plotting 
# plot(mcp)
# Extract coefficients
coefs <- coef(mcp)
p <- length(colnames(X)) # number of predictors

# Create a plot
plot(NA, xlim=c(1, length(mcp$lambda)), ylim=range(coefs), xlab="Lambda Index", ylab="Coefficients"
     , main = "MCP Regularization Path")
matlines(1:length(mcp$lambda), t(coefs[-1, ]), type="l", col=1:p, lty=1 , xlab = "Lambda Index", 
         ylab = "Coefficients")


# this works! 

mc_model2 <- ncvreg(X, y) # MCP default 
# maybe we're having the transpose problem again

# now we can do this same thing with SCAD 
scad <- ncvreg(X, y, penalty = "SCAD") # using the SCAD model now
# testing this 
summary(scad, lambda = 0.1)
coefs_s <- coef(scad)
# now can move on to plotting!
# we will try and plot this in the same way 
plot(NA, xlim = c(1, length(scad$lambda)), ylim=range(coefs_s), xlab="Lambda Index", ylab="Coefficients"
     , main = "SCAD Regularization Path")
matlines(1:length(scad$lambda), t(coefs[-1, ]), type="l", col=1:p, lty=1)
plot(mc_model2, type = "scale", vertical.line = TRUE)
summary(mc_model2)

# it appears we have many 0 coefficients and a sparse model, which, when 
# selecting specific alpha, allows us to see which variables are most 
# strongly correlated with the outcome variable

# we can now try this with the SCAD penalty to observe any differences
mc_model_scad <- ncvreg(X, y, penalty = "SCAD")

# can now observe any differences 
plot(mc_model_scad, type = "scale")

summary(mc_model_scad)

# we can try cross-validating to get around the lambda issue 
cvfit <- cv.ncvreg(X, y)
summary(cvfit)

cvfit_scad <- cv.ncvreg(X, y, penalty = "SCAD")
summary(cvfit_scad)


