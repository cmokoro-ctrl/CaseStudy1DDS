
# CaseStudy1DDS
---
title: "DDSAnalytics: Predicting Employee Attrition"
subtitle: "Case Study 1 - MSDS 6306 Doing Data Science"
author: "Chinedu Okoro"
date: "`r format(Sys.Date(), '%B %d, %Y')`"
output:
  html_document:
    toc: true
    toc_float: true
    toc_depth: 3
    number_sections: true
    theme: flatly
    code_folding: show
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE, fig.align = "center")
library(dplyr)
library(ggplot2)
library(e1071)
library(class)
library(caret)
library(gridExtra)
library(corrplot)
```

# Executive Summary

* **Description of requirements**: 

 DDSAnalytics was asked to identify the drivers of
employee attrition for Frito Lay and build a model to flag at-risk    employees so HR
  can intervene before they leave.
* **Top 3 factors significantly related to attrition**: 
(1) **Overtime** (31.7% vs.
  9.7% attrition)
  
(2) **Monthly Income / Job Level** (lowest earners leave at nearly
  3x the rate of the highest earners)
  
(3) **Job Role** (Sales Representatives
  leave at 45.3%, far above any other role).
  
* **Model performance**: A Naive Bayes classifier achieves cross-validated
  sensitivity ≈ `r round(best_nb$mean_sens*100,0)`% and specificity ≈
  `r round(best_nb$mean_spec*100,0)`%, exceeding the 60%/60% requirement.
  
* **Additional findings**: Employees with no stock options and single employees
  leave at roughly 3x the rate of their peers — both represent retention levers
  that do not require a pay raise.
  
* **Business impact**: Targeting the model's at-risk list with a \$200 incentive
  is estimated to save roughly **\$`r formatC(impact_tbl["mid","savings"], format="d", big.mark=",")`
  per year** versus doing nothing, even under conservative assumptions.

*Video presentation link* : https://smu.zoom.us/rec/share/QHPkMGY9_lKQDSmC_6XNOeBC8ePx-ktUV8_5xDdx0ozb39fTFtOje9gQLGFY_cnS.gWfJwyC1O2C2GQ4s 
Passcode: 1*dQPd=3



# Introduction

DDSAnalytics has been engaged by Frito Lay to identify the factors that drive voluntary
employee turnover (attrition) and to build a model that predicts which employees are
at risk of leaving. Replacing an employee is expensive: the company estimates it costs
between **50% and 400% of an employee's salary** to recruit a replacement, while a
proactive retention incentive costs roughly **$200 per employee**. The goals of this
analysis are to:

1. Identify and provide evidence for the **top three factors** that contribute to attrition.
2. Build a classification model that predicts attrition with **at least 60% sensitivity
   and 60% specificity**.
3. Translate model performance into an estimate of potential cost savings for the business.
4. Surface any other interesting relationships in the data that could inform HR policy.

```{r load-data}
train_raw <- read.csv("CaseStudy1-data.csv", stringsAsFactors = TRUE)
comp_raw  <- read.csv("CaseStudy1CompSet_No_Attrition.csv", stringsAsFactors = TRUE)

dim(train_raw)
sum(is.na(train_raw))
table(train_raw$Attrition)
```

The training set contains `r nrow(train_raw)` employees with `r sum(train_raw$Attrition=="Yes")`
(`r round(mean(train_raw$Attrition=="Yes")*100,1)`%) having left the company, and there is no
missing data. This is a meaningfully **imbalanced** classification problem, which we account
for below when choosing a decision threshold.

# Data Cleaning

Five columns carry no information for modeling: `ID` and `EmployeeNumber` are identifiers,
and `EmployeeCount`, `Over18`, and `StandardHours` are constant for every employee in the
data set. These are dropped. Several right-skewed count/tenure variables
(`MonthlyIncome`, `TotalWorkingYears`, `YearsAtCompany`, `YearsInCurrentRole`,
`YearsWithCurrManager`, `DistanceFromHome`) are log-transformed (`log1p`), which both
stabilizes their variance and better satisfies the Naive Bayes model's within-class
normality assumption for continuous predictors.

```{r clean-data}
clean_data <- function(df) {
  df <- df %>% select(-any_of(c("EmployeeCount", "Over18", "StandardHours")))
  skew_vars <- c("MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
                 "YearsInCurrentRole", "YearsWithCurrManager", "DistanceFromHome")
  for (v in skew_vars) df[[paste0("log_", v)]] <- log1p(df[[v]])
  df
}

train <- clean_data(train_raw)
train$Attrition <- factor(train$Attrition, levels = c("No", "Yes"))

comp <- clean_data(comp_raw)
```

# Exploratory Data Analysis

## Which factors are associated with attrition?

To screen candidate predictors objectively before modeling, every categorical variable
was tested against `Attrition` with a chi-square test of independence (effect size:
Cramer's V), and every numeric variable was tested with a Mann-Whitney U test
(effect size: rank-biserial correlation). The strongest, most significant associations
were:

```{r assoc-table}
cat_cols <- c("OverTime","JobRole","MaritalStatus","Department","BusinessTravel",
              "EducationField","Gender")
num_cols <- setdiff(names(train), c(cat_cols, "Attrition", "ID", "EmployeeNumber"))
num_cols <- num_cols[sapply(train[num_cols], is.numeric)]

cat_assoc <- lapply(cat_cols, function(c) {
  tab <- table(train[[c]], train$Attrition)
  test <- suppressWarnings(chisq.test(tab))
  n <- sum(tab); k <- min(dim(tab)) - 1
  v <- sqrt((test$statistic / n) / k)
  data.frame(variable = c, type = "categorical", p_value = test$p.value, effect_size = unname(v))
}) %>% bind_rows()

num_assoc <- lapply(num_cols, function(c) {
  yes <- train[[c]][train$Attrition == "Yes"]
  no  <- train[[c]][train$Attrition == "No"]
  test <- wilcox.test(yes, no)
  n1 <- length(yes); n2 <- length(no)
  rb <- 1 - (2 * test$statistic) / (n1 * n2)
  data.frame(variable = c, type = "numeric", p_value = test$p.value, effect_size = unname(rb))
}) %>% bind_rows()

assoc <- bind_rows(cat_assoc, num_assoc) %>% arrange(p_value)
knitr::kable(head(assoc, 12), digits = c(NA,NA,5,3), caption = "Strongest predictors of attrition (by p-value)")
```

## Top 3 factors related to attrition

Based on statistical significance, effect size, and business interpretability
(avoiding variables that are just proxies for one another), the three strongest and
most actionable drivers of attrition are:

### 1. Overtime

```{r eda-overtime, fig.width=5, fig.height=3.5}
ggplot(train, aes(x = OverTime, fill = Attrition)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Attrition Rate by Overtime Status", y = "Proportion", x = "Works Overtime") +
  theme_minimal()
```

Employees who work overtime leave at **31.7%**, more than **3x** the rate of those who
do not (**9.7%**). This is the single strongest, most distinct predictor in the data
set (Cramer's V = 0.27, p < .0001) and is not simply a proxy for another variable.

### 2. Monthly Income (compensation / job level)

```{r eda-income, fig.width=5, fig.height=3.5}
ggplot(train, aes(x = Attrition, y = MonthlyIncome, fill = Attrition)) +
  geom_boxplot() +
  labs(title = "Monthly Income by Attrition Status", y = "Monthly Income ($)") +
  theme_minimal() + theme(legend.position = "none")
```

Employees who leave earn a median of **$3,171/month** versus **$5,209/month** for
those who stay — roughly 40% less. `MonthlyIncome` is highly correlated with
`JobLevel` (r = 0.95), so these should be interpreted as the same underlying
"compensation / seniority" signal. The lowest income quartile has a **28.9%**
attrition rate versus **10.1%** in the top quartile.

### 3. Job Role

```{r eda-jobrole, fig.width=7, fig.height=4}
role_rate <- train %>% group_by(JobRole) %>%
  summarise(attr_rate = mean(Attrition == "Yes"), n = n()) %>%
  arrange(desc(attr_rate))

ggplot(role_rate, aes(x = reorder(JobRole, attr_rate), y = attr_rate)) +
  geom_col(fill = "#D9534F") +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Attrition Rate by Job Role", x = NULL, y = "Attrition Rate") +
  theme_minimal()
```

Attrition is heavily concentrated in a few roles. **Sales Representatives** leave at
an alarming **45.3%** rate, and **Human Resources** and **Laboratory Technician**
roles are also elevated (19–22%), while **Research Director** and
**Manufacturing Director** are below 3%. This points to role-specific drivers
(workload, travel, compensation structure) rather than a company-wide problem.

## Other interesting findings

```{r eda-other, fig.width=9, fig.height=3.5}
p1 <- ggplot(train, aes(x = factor(StockOptionLevel), fill = Attrition)) +
  geom_bar(position = "fill") + scale_y_continuous(labels = scales::percent) +
  labs(title = "By Stock Option Level", x = "Stock Option Level", y = NULL) + theme_minimal()

p2 <- ggplot(train, aes(x = MaritalStatus, fill = Attrition)) +
  geom_bar(position = "fill") + scale_y_continuous(labels = scales::percent) +
  labs(title = "By Marital Status", x = NULL, y = NULL) + theme_minimal()

grid.arrange(p1, p2, ncol = 2)
```

* **Stock options matter**: employees with **no** stock options (level 0) leave at
  **25.9%**, versus **7–8%** for levels 1–2.
  
* **Single employees** leave more than twice as often as married employees (26.0%
  vs. 14.1%) and four times as often as divorced employees (6.3%).
* Tenure-related variables (`YearsAtCompany`, `YearsInCurrentRole`,
  `YearsWithCurrManager`, `TotalWorkingYears`, `Age`) are all strongly correlated
  with each other and with attrition — newer employees are consistently more likely
  to leave, which is consistent with the Job Role and Income findings above.

```{r eda-corr, fig.width=6, fig.height=6}
corr_vars <- c("Age","MonthlyIncome","JobLevel","TotalWorkingYears","YearsAtCompany",
               "YearsInCurrentRole","YearsWithCurrManager","StockOptionLevel")
corrplot(cor(train[corr_vars]), method = "color", type = "upper",
         addCoef.col = "black", number.cex = 0.7, tl.cex = 0.8)
```

# Predictive Modeling

The business requires a classification model with **at least 60% sensitivity and 60%
specificity**. Given the moderate sample size (870 employees, 140 attrition cases),
performance was evaluated with **repeated 10-fold cross-validation** rather than a
single train/test split, which can be misleadingly optimistic or pessimistic by
chance. Two classifiers taught in this course were compared: **Naive Bayes** and
**k-Nearest Neighbors (kNN)**.

```{r model-setup}
feat <- c("OverTime","JobRole","MaritalStatus","Department","BusinessTravel",
          "StockOptionLevel","log_TotalWorkingYears","log_MonthlyIncome","JobLevel",
          "log_YearsAtCompany","log_YearsInCurrentRole","JobInvolvement",
          "log_YearsWithCurrManager","Age","JobSatisfaction","log_DistanceFromHome",
          "WorkLifeBalance","EnvironmentSatisfaction")

model_df <- train[, c(feat, "Attrition")]

sens_spec <- function(prob, truth, thresh) {
  pred <- factor(ifelse(prob >= thresh, "Yes", "No"), levels = c("No", "Yes"))
  cm <- caret::confusionMatrix(pred, truth, positive = "Yes")
  c(sens = unname(cm$byClass["Sensitivity"]), spec = unname(cm$byClass["Specificity"]))
}
```

The model uses `r length(feat)` predictors (more than the three headline factors
above — the assignment only requires that we *identify* the top three, not that the
model be restricted to them). Every variable included was significant at p < .05 in
the screening step above.

## Naive Bayes (primary model)

```{r nb-cv}
set.seed(6306)
thresholds <- seq(0.10, 0.60, by = 0.01)
n_reps <- 5
sens_mat <- matrix(NA, nrow = 0, ncol = length(thresholds))
spec_mat <- matrix(NA, nrow = 0, ncol = length(thresholds))

for (r in 1:n_reps) {
  folds <- createFolds(model_df$Attrition, k = 10)
  for (f in folds) {
    tr <- model_df[-f, ]; te <- model_df[f, ]
    nb_fit <- naiveBayes(Attrition ~ ., data = tr, laplace = 1)
    prob <- predict(nb_fit, newdata = te, type = "raw")[, "Yes"]
    row_s <- sapply(thresholds, function(t) sens_spec(prob, te$Attrition, t)["sens"])
    row_p <- sapply(thresholds, function(t) sens_spec(prob, te$Attrition, t)["spec"])
    sens_mat <- rbind(sens_mat, row_s)
    spec_mat <- rbind(spec_mat, row_p)
  }
}

cv_results <- data.frame(thresh = thresholds,
                          mean_sens = colMeans(sens_mat),
                          mean_spec = colMeans(spec_mat))
cv_results$min_both <- pmin(cv_results$mean_sens, cv_results$mean_spec)
best_nb <- cv_results[which.max(cv_results$min_both), ]
knitr::kable(best_nb, digits = 3, row.names = FALSE,
             caption = "Naive Bayes: best threshold from 5x repeated 10-fold CV")
```

```{r nb-cv-plot, fig.width=6, fig.height=3.5}
ggplot(cv_results, aes(x = thresh)) +
  geom_line(aes(y = mean_sens, color = "Sensitivity")) +
  geom_line(aes(y = mean_spec, color = "Specificity")) +
  geom_hline(yintercept = 0.60, linetype = "dashed", color = "gray40") +
  geom_vline(xintercept = best_nb$thresh, linetype = "dotted") +
  labs(title = "Naive Bayes: Sensitivity/Specificity vs. Threshold (CV average)",
       x = "Classification threshold", y = "Rate", color = NULL) +
  theme_minimal()
```

At threshold `r round(best_nb$thresh,2)`, Naive Bayes achieves cross-validated
**sensitivity = `r round(best_nb$mean_sens,3)`** and
**specificity = `r round(best_nb$mean_spec,3)`**, comfortably above the 60/60 bar.

## k-Nearest Neighbors (comparison model)

```{r knn-cv}
num_feat <- feat[sapply(model_df[feat], is.numeric)]
cat_feat <- setdiff(feat, num_feat)
dummies <- dummyVars(~ ., data = model_df[, cat_feat])
cat_mat <- predict(dummies, newdata = model_df[, cat_feat, drop = FALSE])
num_mat <- scale(model_df[, num_feat])
knn_X <- cbind(num_mat, cat_mat)
knn_y <- model_df$Attrition

set.seed(6306)
folds <- createFolds(knn_y, k = 10)
k_grid <- c(15, 21, 27, 31)
knn_thresh <- seq(0.08, 0.35, by = 0.01)
knn_summary <- list()

for (k in k_grid) {
  sens_mat <- matrix(NA, nrow = 0, ncol = length(knn_thresh))
  spec_mat <- matrix(NA, nrow = 0, ncol = length(knn_thresh))
  for (f in folds) {
    fit <- knn(train = knn_X[-f, ], test = knn_X[f, ], cl = knn_y[-f], k = k, prob = TRUE)
    winprob <- attr(fit, "prob")
    prob_yes <- ifelse(fit == "Yes", winprob, 1 - winprob)
    row_s <- sapply(knn_thresh, function(t) sens_spec(prob_yes, knn_y[f], t)["sens"])
    row_p <- sapply(knn_thresh, function(t) sens_spec(prob_yes, knn_y[f], t)["spec"])
    sens_mat <- rbind(sens_mat, row_s); spec_mat <- rbind(spec_mat, row_p)
  }
  res <- data.frame(k = k, thresh = knn_thresh, mean_sens = colMeans(sens_mat), mean_spec = colMeans(spec_mat))
  res$min_both <- pmin(res$mean_sens, res$mean_spec)
  knn_summary[[as.character(k)]] <- res[which.max(res$min_both), ]
}
knn_summary <- bind_rows(knn_summary)
knitr::kable(knn_summary, digits = 3, row.names = FALSE,
             caption = "kNN: best threshold per k (10-fold CV)")

best_knn <- knn_summary[which.max(knn_summary$min_both), ]
```

The best kNN configuration (k = `r best_knn$k`, threshold = `r best_knn$thresh`) reaches
sensitivity = `r round(best_knn$mean_sens,3)` and specificity = `r round(best_knn$mean_spec,3)`,
also clearing 60/60, but with a slightly worse balance than Naive Bayes.

## Model selection

```{r model-compare}
compare_tbl <- data.frame(
  Model = c("Naive Bayes", paste0("kNN (k=", best_knn$k, ")")),
  Threshold = c(best_nb$thresh, best_knn$thresh),
  Sensitivity = c(best_nb$mean_sens, best_knn$mean_sens),
  Specificity = c(best_nb$mean_spec, best_knn$mean_spec)
)
knitr::kable(compare_tbl, digits = 3)
```

**Naive Bayes is selected as the final model** for the competition predictions: it
clears the 60/60 requirement with more margin and a better sensitivity/specificity
balance, is computationally simpler to deploy and re-threshold, and its
per-predictor likelihoods are easy to explain to a non-technical audience.

## Final model fit (on all training data)

```{r final-fit}
final_model <- naiveBayes(Attrition ~ ., data = model_df, laplace = 1)
final_thresh <- best_nb$thresh
```

The final Naive Bayes model is refit on the **full 870-employee training set** (using
all available labeled data, rather than a 70/30 split) before being applied to the
300-employee competition set, to maximize the information the model has access to.

# Estimated Financial Impact

Using the cross-validated confusion matrix rates, scaled to the full population of
`r nrow(train)` employees (`r sum(train$Attrition=="Yes")` actual leavers):

```{r financial-impact}
avg_salary <- mean(train$MonthlyIncome) * 12
n <- nrow(train)
n_leavers <- sum(train$Attrition == "Yes")
n_stayers <- n - n_leavers

sens <- best_nb$mean_sens; spec <- best_nb$mean_spec
TP <- round(n_leavers * sens); FN <- n_leavers - TP
TN <- round(n_stayers * spec); FP <- n_stayers - TN

incentive <- 200
repl_rates <- c(low = 0.5, mid = 1.5, high = 4.0)

impact <- sapply(repl_rates, function(r) {
  baseline <- n_leavers * avg_salary * r
  with_model <- TP * incentive + FN * avg_salary * r + FP * incentive
  c(baseline = baseline, with_model = with_model, savings = baseline - with_model)
})
impact_tbl <- as.data.frame(t(impact))
impact_tbl$scenario <- rownames(impact_tbl)
knitr::kable(impact_tbl[, c("scenario","baseline","with_model","savings")],
             format.args = list(big.mark = ","), digits = 0,
             caption = "Estimated annual turnover cost: no model vs. model-targeted $200 incentive")
```


# Competition Set Predictions

```{r competition-predictions}
comp_feat <- comp[, feat]
comp_prob <- predict(final_model, newdata = comp_feat, type = "raw")[, "Yes"]
comp_pred <- ifelse(comp_prob >= final_thresh, "Yes", "No")

submission <- data.frame(ID = comp$ID, Attrition = comp_pred)
write.csv(submission, "Case1PredictionsOkoro Attrition.csv", row.names = FALSE, quote = FALSE)

table(submission$Attrition)
head(submission)
```



