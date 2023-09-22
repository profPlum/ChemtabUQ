library(tidyverse)
source('~/.Rprofile')

########################## GLMNET Helper Functions: ##########################

glmnet_R2 = function(glmnet_cv_out, s='lambda.1se') {
  ids = list(lambda.min=glmnet_cv_out$index[[1]], lambda.1se=glmnet_cv_out$index[[2]])
  R_Squared_train = glmnet_cv_out$glmnet.fit$dev.ratio[[ ids[[s]] ]]
  return(R_Squared_train)
}

# returns coefs as named vector (like we expect)
coef.cv.glmnet = function(cv, s='lambda.1se', ...) {
  lm_coefs_raw = glmnet::coef.glmnet(cv, s=s, ...)
  lm_coefs = as.vector(lm_coefs_raw)
  names(lm_coefs) = rownames(lm_coefs_raw)
  return(lm_coefs)
}

glmnet=function(formula, data, ...) #TODO: figure out corresponding method for predict() which can reuse formula + new df flexibly
  glmnet::cv.glmnet(as.matrix(model.matrix(formula, data=data)), y=data[[ all.vars(formula)[[1]] ]], intercept=F, ...)
  # if user requests it intercept will implicitly be included by formula

##############################################################################

# Params, ideally from CLI...
n_PCs = 25
variance_weighted = F
Chemtab_fn = '~/Downloads/TChem_collated.csv.gz'
#Chemtab_fn <- file.choose()
#Chemtab_fn = commandArgs()[[1]]

Chemtab_data = read_csv(Chemtab_fn)
mass_frac_data = Chemtab_data %>% select(starts_with('Yi'))
souspec_data = Chemtab_data %>% select(starts_with('souspec'))

# I forget why but a long time ago there was some kind of bug/numerical 
# instability caused by non-regularized zmix lm fit. The problem was resolved
# by using lasso regularization on the fit. Hence using glmnet.
zmix_lm = glmnet(zmix~.-1, data=cbind(zmix=Chemtab_data$zmix, mass_frac_data))#, lambda=all_lambdas)
stopifnot(glmnet_R2(zmix_lm)>=0.99)
cat('Zmix lasso-lm coefs: ')
print(coef(zmix_lm)[-1])

# Verified that removing centering doesn't effect reconstruction loss!!
# However removing scaling does indeed negatively effect it
# (unless we want to use mass variance)
mass_PCA = prcomp(mass_frac_data, scale.=!variance_weighted, center=F, rank=n_PCs)
rotation = mass_PCA$rotation
if (!variance_weighted) rotation = diag(1/mass_PCA$scale)%*%mass_PCA$rotation # emb scaling
stopifnot(all.equal(as.matrix(mass_frac_data)%*%rotation, mass_PCA$x))
stopifnot(names(coef(zmix_lm)[-1])==rownames(rotation))
rownames(rotation) = colnames(mass_frac_data)
colnames(rotation) = paste0('CPV_', 1:n_PCs-1) # colnames should match exactly the format from V1
rotation = cbind(zmix=coef(zmix_lm)[-1], rotation) # colnames should match exactly the format from V1
View(rotation[1:5, 1:5])

# NOTE: apparently using linear models here has a noticable decrease on R2 (though slight), so we'll avoid it
# rotation = fit_linear_transform(mass_frac_data, cbind(Chemtab_data$zmix, mass_PCA$x))
Q_rot = rotation %>% qr() %>% qr.Q() # doesn't effect reconstruction loss!
dimnames(Q_rot)=dimnames(rotation)

# It flips the sign of the zmix weights & normalizes them, we flip sign back but keep it normalized
Q_rot = Q_rot*cor(Q_rot[,1], rotation[,1]) # this correlation should be either 1 or -1 & indicates a sign flip
stopifnot(all.equal(Q_rot[,1], rotation[,1]/norm(rotation[,1], type='2')))
# confirm that first CPV is still (porportional to) Zmix

# IMPORTANT: It's OK that zmix is correlated with CPVs!!
# correlation!=W_matrix orthogonality (cor depends on mass_frac data)
mass_PCs = as.matrix(mass_frac_data)%*%Q_rot
R2 = get_explained_var(mass_PCs, mass_frac_data, var_weighted=variance_weighted)
cat('mass_PCs --> mass_frac_data, R2: ', R2, '\n')
stopifnot(R2>=0.90)
cat('range(mass_PCs): ', range(mass_PCs), '\n')
Q_rot = Q_rot[sort(rownames(Q_rot)),]
write.csv(Q_rot, file=paste0('Q_rot', ifelse(variance_weighted, '_MassR2', ''),'.csv.gz'))

####################### Augment Original Dataset with CPVs + CPV_sources #######################

mass_PCs = as.matrix(mass_frac_data)%*%Q_rot %>% as_tibble()
colnames(mass_PCs) = colnames(mass_PCs) %>% paste0('mass_', .)
CPV_sources = as.matrix(souspec_data)%*%Q_rot %>% as_tibble()
colnames(CPV_sources) = colnames(CPV_sources) %>% paste0('source_', .)

Chemtab_data = cbind(Chemtab_data, mass_PCs, CPV_sources)
write.csv(Chemtab_data, file=paste0('Chemtab_data', ifelse(variance_weighted, '_MassR2', ''),'.csv.gz'))

###############################################################################################
