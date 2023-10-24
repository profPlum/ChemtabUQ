library(tidyverse)

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
  names(lm_coefs) = gsub('`', '', rownames(lm_coefs_raw))
  return(lm_coefs)
}

glmnet=function(formula, data, ...) #TODO: figure out corresponding method for predict() which can reuse formula + new df flexibly
  glmnet::cv.glmnet(as.matrix(model.matrix(formula, data=data)), y=data[[ all.vars(formula)[[1]] ]], intercept=F, ...)
  # if user requests it intercept will implicitly be included by formula

##############################################################################

# Accept n_PCs from env variable!
n_PCs=as.integer(Sys.getenv()['N_CPVS'])
if (is.na(n_PCs)) n_PCs = 25 # default is 25 (1-to-1)
cat('n_CPVs: ', n_PCs, '(change via N_CPVS env var)\n')

#Chemtab_fn = '~/Downloads/TChem_collated.csv.gz'
#Chemtab_fn = '~/Downloads/Data/wax_master.csv'
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

# I forget why but a long time ago there was some kind of bug/numerical 
# instability caused by non-regularized zmix lm fit. The problem was resolved
# by using lasso regularization on the fit. Hence using glmnet.
zmix_lm = glmnet(zmix~.-1, data=cbind(zmix=Chemtab_data$zmix, mass_frac_data))
stopifnot(glmnet_R2(zmix_lm)>=0.99)
cat('Zmix lasso-lm coefs: ')
print(coef(zmix_lm)[-1])

export_CPVs_and_rotation = function(variance_weighted=T) {
  # Verified that removing centering doesn't effect reconstruction loss!! 9/21/23
  # However removing scaling does indeed negatively effect it
  # (unless we want to use mass variance weights...)
  mass_PCA = prcomp(mass_frac_data, scale.=!variance_weighted, center=F, rank=n_PCs)
  rotation = mass_PCA$rotation
  if (!variance_weighted) rotation = diag(1/mass_PCA$scale)%*%mass_PCA$rotation # emb scaling
  stopifnot(all.equal(as.matrix(mass_frac_data)%*%rotation, mass_PCA$x))
  stopifnot(names(coef(zmix_lm)[-1])==rownames(rotation))
  rownames(rotation) = colnames(mass_frac_data)
  colnames(rotation) = paste0('CPV_PC_', 1:n_PCs-1) # colnames renamed from V1 for clarity & for matching with 'like' in pandas
  rotation = cbind(CPV_zmix=coef(zmix_lm)[-1], rotation) # ^ I've confirmed that ablate code doesn't rely on column names anyways...
  #View(rotation[1:5, 1:5])

  # NOTE: apparently using linear models here has a noticable decrease on R2 (though slight), so we'll avoid it
  # rotation = fit_linear_transform(mass_frac_data, cbind(Chemtab_data$zmix, mass_PCA$x))
  Q_rot = rotation %>% qr() %>% qr.Q() # doesn't effect reconstruction loss!
  dimnames(Q_rot)=dimnames(rotation)

  # It flips the sign of the zmix weights & normalizes them, we flip sign back but keep it normalized
  Q_rot = Q_rot*cor(Q_rot[,1], rotation[,1]) # this correlation should be either 1 or -1 & indicates a sign flip
  stopifnot(all.equal(Q_rot[,1], rotation[,1]/norm(as.matrix(rotation[,1]), type='2')))
  # Confirm that first CPV is still (proportional to) Zmix. Also kenny confirmed proportional to is enough.
  # IMPORTANT: It's OK that zmix is correlated with CPVs!!
  # correlation!=W_matrix orthogonality (cor depends on mass_frac data)
  
  ####################### Augment Original Dataset with CPVs + CPV_sources #######################

  stopifnot(sub('souspec', 'Yi', colnames(souspec_data))==rownames(Q_rot))
  CPV_sources = as.matrix(souspec_data)%*%Q_rot %>% as_tibble()
  colnames(CPV_sources) = colnames(CPV_sources) %>% paste0('source_', .)
  
  stopifnot(colnames(mass_frac_data)==rownames(Q_rot))
  mass_PCs = as.matrix(mass_frac_data)%*%Q_rot %>% as_tibble()
  colnames(mass_PCs) = colnames(mass_PCs) %>% paste0('mass_', .)

  ################################ Sanity Checks ################################################
  
  R2 = get_explained_var(mass_PCs, mass_frac_data, var_weighted=variance_weighted)
  cat('mass_PCs --> mass_frac_data, R2: ', R2, '\n')
  stopifnot(R2>=0.95)
  cat('range(mass_PCs): ', range(mass_PCs), '\n')
  
  ################################# Write Files #################################################
  
  # we don't use slice_sample() anymore for 2 reasons: it makes finding ideal seed across different CPV datasets very difficult,
  # it is already done both inside ChemtabUQ.py & Collate_Ablate_Data.R (both of these places do not interfere though)
  Chemtab_data = cbind(Chemtab_data, mass_PCs, CPV_sources) %>% as_tibble #%>% slice_sample(prop=1)
  write.csv(Chemtab_data, file=paste0('TChem+CPVs+Zmix', ifelse(variance_weighted, '_MassR2', ''),'.csv.gz'))

  # don't sort the order of Q_rot rownames! Existing order is important as it reflects order in mech file!
  rownames(Q_rot) = sub('Yi', '', rownames(Q_rot)) # we need to strip the 'Yi' prefix b/c ablate requires it (& only ablate will use this!)
  write.csv(Q_rot, file=paste0('Q_rot', ifelse(variance_weighted, '_MassR2', ''),'.csv.gz'))
  
  ###############################################################################################
}

# We're not sure 'which is better', but having both is good for experimentation!
export_CPVs_and_rotation(variance_weighted=T)
export_CPVs_and_rotation(variance_weighted=F)
