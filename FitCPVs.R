library(tidyverse)
source('~/.Rprofile')

########################## Helper Functions: ##########################

# Everything is verified except whether it is ok to ignore bias: 8/2/22
fit_linear_transform = function(X_df, Y_df) {
  # verified to work 8/2/22 (checks that data frames are all numeric)
  stopifnot(is.numeric(as.matrix(X_df)))
  stopifnot(is.numeric(as.matrix(Y_df)))

  rotation_matrix = NULL
  for (j in 1:ncol(Y_df)) {
    # TODO: verify that it is ok to ignore bias here?
    model = lm(as.matrix(Y_df)[,j]~.-1, data=X_df) 
  
    print(summary(model))
    rotation_matrix = cbind(rotation_matrix, coef(model))
  }
  colnames(rotation_matrix) = colnames(Y_df)
  rownames(rotation_matrix) = gsub('`', '', rownames(rotation_matrix))

  
  # print R^2 of entire rotation matrix
  R2 = 1-sum(apply((as.matrix(X_df)%*%rotation_matrix-Y_df)**2, -1, mean)/apply(Y_df, -1, var))
  cat('R2 of linear transform fit: ', R2)

  return(rotation_matrix)
}

glmnet_R2 = function(glmnet_cv_out, s='lambda.min') {
  ids = list(lambda.min=glmnet_cv_out$index[[1]], lambda.1se=glmnet_cv_out$index[[2]])
  R_Squared_train = glmnet_cv_out$glmnet.fit$dev.ratio[[ ids[[s]] ]]
  return(R_Squared_train)
}

# returns coefs as named vector (like we expect)
coef.cv.glmnet = function(cv, s='lambda.min', ...) {
  lm_coefs_raw = glmnet::coef.glmnet(cv, s=s, ...)
  lm_coefs = as.vector(lm_coefs_raw)
  names(lm_coefs) = gsub('`', '', rownames(lm_coefs_raw))
  return(lm_coefs)
}

glmnet=function(formula, data, ...) #TODO: figure out corresponding method for predict() which can reuse formula + new df flexibly
  glmnet::cv.glmnet(as.matrix(model.matrix(formula, data=data)), y=data[[ all.vars(formula)[[1]] ]], intercept=F, ...)
  # if user requests it intercept will implicitly be included by formula

# NOTE: you can pass lm=glm for glm type model! (glm doesn't show R^2 though...)
fit_significant_lm = function(formula, data, significance_level=0.05,
                              lm_fit=lm, ...) {
  # NOTE: this is how you get p_values!
  # https://stackoverflow.com/questions/16153497/selecting-the-statistically-significant-variables-in-an-r-glm-model
  p.vals = function(m) {
    stopifnot('lm' %in% class(m))
    return(summary(m)$coeff[-1,4])
  }

  # Refine lm's x.vars until only significant predictors remain
  str_formula = as.character(formula)[-1] # index [1] is the tilde
  y.var <- str_formula[[1]]
  x.vars <- str_formula[[2]]
  print(x.vars)
  sign.lm = lm_fit(formula, data=data, ...)
  while(max(p.vals(sign.lm))>significance_level) { # while insignificant predictors remain
    irrelevant.x <- names(p.vals(sign.lm))[which.max(p.vals(sign.lm))]
    x.vars <- paste0(x.vars, '-', irrelevant.x)
    print(x.vars)
    sig.formula <- as.formula(paste(y.var, "~", x.vars))
    sign.lm = lm_fit(sig.formula, data=data, ...)
  }
  print(summary(sign.lm))

  return(sign.lm)
}

##############################################################################

# Accept n_PCs from env variable!
n_PCs=as.integer(Sys.getenv()['N_CPVS'])
if (is.na(n_PCs)) n_PCs = 25 # default is 25 which is more than enough, you can cut short by using regex pattern
cat('N_CPVs: ', n_PCs, '(change via N_CPVS env var)\n')

use_QR=as.logical(Sys.getenv()['QR'])
if (is.na(use_QR)) use_QR=FALSE # the original motivation for QR should be irrelevant now...
cat('QR: ', use_QR, '(change via QR=T/F env var)\n')

Chemtab_fn = commandArgs(trailingOnly = T)[[1]]
cat('Chemtab_fn: ', Chemtab_fn, '\n')
cat('CDing to data directory.\n') # AFTER loading csv...
Sys.sleep(1)

Chemtab_data = read_csv(Chemtab_fn) %>% mutate(souspecAR=0)
if ('Zmix' %in% colnames(Chemtab_data)) Chemtab_data = Chemtab_data %>% rename(zmix=Zmix)
mass_frac_data = Chemtab_data %>% select(starts_with('Yi'))
souspec_data = Chemtab_data %>% select(starts_with('souspec'))

# now we CD (so outputs are in data directory)
setwd(dirname(Chemtab_fn))

# # I forget why but a long time ago there was some kind of bug/numerical
# # instability caused by non-regularized zmix lm fit. The problem was resolved
# # by using lasso regularization on the fit. Hence using glmnet.
# training_data <- cbind(zmix=Chemtab_data$zmix, mass_frac_data)
# zmix_lm = glmnet(zmix~.-1, data=training_data)
# stopifnot(glmnet_R2(zmix_lm, s='lambda.min')>=0.999)
# cat('Zmix lasso-lm coefs: ')
# zmix_coefs=coef(zmix_lm, s='lambda.min')[-1]
# print(zmix_coefs)

# NOTE: I've confirmed that using lasso lm although convenient actually gives
# really bad zmix source term (like 60k at times, despite R^2>0.99), which means it isn't viable...
train_data = cbind(zmix=Chemtab_data$zmix, mass_frac_data) |> slice_sample(prop=0.8)
holdout = Chemtab_data |> anti_join(train_data)
zmix_lm = fit_significant_lm(zmix~.-1-YiAR, data=train_data)
preds = predict(zmix_lm, newdata=holdout)
holdout_R2=R2(holdout$zmix, preds)
cat('Zmix_lm holdout R^2=', holdout_R2)
stopifnot(holdout_R2>=0.999)
stopifnot(summary(zmix_lm)$adj.r.squared>=0.999)
zmix_coefs= coef(zmix_lm)
names(zmix_coefs)=gsub('`', '', names(zmix_coefs))
excluded_Yis = setdiff(colnames(mass_frac_data), names(zmix_coefs))
zmix_coefs[excluded_Yis]=0
zmix_coefs=zmix_coefs[colnames(mass_frac_data)] # sort
stopifnot(max(abs(zmix_coefs)) < 50)

save(zmix_lm, file='zmix_lm.RData')

export_CPVs_and_rotation = function(use_QR=F) {
  # NOTE: ONLY variance weighted PCA makes sense... Because if you apply the scaling matrix then 1 of 2 things
  # happen: either you have: an enormous matrix which is not at all orthonormal (e.g. col norm~=1e20, hence bounds
  # of proof doesn't apply), or you then apply QR which removes the scaling anyhow since it enforces orthonormality. 
  variance_weighted=T # NOTE: ONLY variance weighted PCA makes sense...
  stopifnot(variance_weighted)

  # Verified that removing centering doesn't effect reconstruction loss!! 9/21/23
  # However removing scaling does indeed negatively effect it
  # (unless we want to use mass variance weights...)
  mass_PCA = prcomp(mass_frac_data, scale.=!variance_weighted, center=F, rank=n_PCs)
  #rotation=fit_linear_transform(mass_frac_data, mass_PCA$x)
  # NOTE: apparently using linear models here has a noticable decrease on R2 (though slight)

  rotation = mass_PCA$rotation
  if (!variance_weighted) rotation = diag(1/mass_PCA$scale)%*%mass_PCA$rotation # emb scaling
  stopifnot(all.equal(as.matrix(mass_frac_data)%*%rotation, mass_PCA$x))
  stopifnot(all(names(zmix_coefs)==rownames(rotation)))
  rownames(rotation) = colnames(mass_frac_data)
  colnames(rotation) = paste0('CPV_PC_', 1:n_PCs-1) # colnames renamed from V1 for clarity & for matching with 'like' in pandas
  zmix_coef_norm=norm(as.matrix(zmix_coefs), type='2')
  rotation = cbind(CPV_zmix=zmix_coefs/zmix_coef_norm, rotation) # ^ I've confirmed that ablate code doesn't rely on column names anyways...

  if (use_QR) {
    Q_rot = rotation %>% qr() %>% qr.Q() # doesn't effect reconstruction loss!
    dimnames(Q_rot)=dimnames(rotation)

    # It flips the sign of the zmix weights & normalizes them, we flip sign back but keep it normalized
    Q_rot = Q_rot*cor(Q_rot[,1], rotation[,1]) # this correlation should be either 1 or -1 & indicates a sign flip
    stopifnot(all.equal(Q_rot[,1], rotation[,1]/norm(as.matrix(rotation[,1]), type='2')))
    # Confirm that first CPV is still (proportional to) Zmix. Also kenny confirmed proportional to is enough.
    # IMPORTANT: It's OK that zmix is correlated with CPVs!!
    # correlation!=W_matrix orthogonality (cor depends on mass_frac data)
  } else {
    Q_rot=rotation
  }

  ####################### Augment Original Dataset with CPVs + CPV_sources #######################

  stopifnot(sub('souspec', 'Yi', colnames(souspec_data))==rownames(Q_rot))
  CPV_sources = as.matrix(souspec_data)%*%Q_rot %>% as_tibble()
  colnames(CPV_sources) = colnames(CPV_sources) %>% paste0('source_', .)
  stopifnot(max(abs(CPV_sources$source_CPV_zmix))<1e-6) # should be approximately 0
  
  stopifnot(colnames(mass_frac_data)==rownames(Q_rot))
  mass_PCs = as.matrix(mass_frac_data)%*%Q_rot %>% as_tibble()
  colnames(mass_PCs) = colnames(mass_PCs) %>% paste0('mass_', .)

  ################################ Sanity Checks ################################################
  
  R2 = get_explained_var(mass_PCs, mass_frac_data, var_weighted=variance_weighted)
  cat('mass_PCs --> mass_frac_data, R2: ', R2, '\n')
  stopifnot(R2>=0.98)
  cat('range(mass_PCs): ', range(mass_PCs), '\n')
  
  ################################# Write Files #################################################
  
  # we don't use slice_sample() anymore for 2 reasons: it makes finding ideal seed across different CPV datasets very difficult,
  # it is already done both inside ChemtabUQ.py & Collate_Ablate_Data.R (both of these places do not interfere though)
  Chemtab_data = cbind(Chemtab_data, mass_PCs, CPV_sources) %>% as_tibble %>% arrange(time) # %>% slice_sample(prop=1)
  write.csv(Chemtab_data, row.names=F, file=paste0('TChem+CPVs+Zmix', ifelse(use_QR, '_QR', ''), ifelse(variance_weighted, '_MassR2', ''),'.csv.gz'))

  # don't sort the order of Q_rot rownames! Existing order is important as it reflects order in mech file!
  rownames(Q_rot) = sub('Yi', '', rownames(Q_rot)) # we need to strip the 'Yi' prefix b/c ablate requires it (& only ablate will use this!)
  write.csv(Q_rot, file=paste0(ifelse(use_QR, 'Q_rot', 'rot'), ifelse(variance_weighted, '_MassR2', ''),'.csv.gz'))
  
  return(lst(Chemtab_data, Q_rot))
  ###############################################################################################
}

result=export_CPVs_and_rotation(use_QR=use_QR)
