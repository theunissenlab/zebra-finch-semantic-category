# Mixed effects binomial in R for lesion paper

library(lme4)
library(stringr)

# Read the data frame.
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
fileIn <- '../results/resultsBirdR.csv'
post_counts <- read.csv(fileIn)


# Base Model
null_model <- glmer('cbind(Int, NoInt) ~ 1 + ( 1 |Bird)', data = post_counts, family = binomial)
base_model <- glmer('cbind(Int, NoInt) ~ Re + ((1 + Re)|Bird)', data = post_counts, family = binomial)
callType_model <- glmer('cbind(Int, NoInt) ~ Re*Call + ((1 + Re)|Bird)', data = post_counts, family = binomial)
sex_model <- glmer('cbind(Int, NoInt) ~ Re*Sex + ((1 + Re)|Bird)', data = post_counts, family = binomial)

# This is significant because Re changes interruption rates
anova(null_model, base_model)

# Test for sex differences.
anova(sex_model, base_model)

# Test for call differences
anova(callType_model, base_model)


