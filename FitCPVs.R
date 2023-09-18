library(tidyverse)
source('~/.Rprofile')

# Params, ideally from CLI...
n_PCs = 25
variance_weighted = F
#Chemtab_fn <- file.choose()
Chemtab_fn = commandArgs()[[1]]

Chemtab_data = read_csv(Chemtab_fn)
mass_frac_data = Chemtab_data %>% select(starts_with('Yi'))
souspec_data = Chemtab_data %>% select(starts_with('souspec'))

# Verified that removing centering doesn't effect reconstruction loss!!
# However removing scaling does indeed negatively effect it
# (unless we want to use mass variance)
mass_PCA = prcomp(mass_frac_data, scale.=!variance_weighted, center=F, rank=n_PCs)
#full_rotation = mass_PCA$rotation
#full_rotation = diag(1/mass_PCA$scale)%*%mass_PCA$rotation
full_rotation = fit_linear_transform(mass_frac_data, mass_PCA$x)
rotation = full_rotation[,1:n_PCs]
Q_rot = rotation %>% qr() %>% qr.Q() # doesn't effect reconstruction loss!
rownames(Q_rot) = colnames(mass_frac_data)
colnames(Q_rot) = paste0('CPV_', 1:ncol(Q_rot))
mass_PCs = as.matrix(mass_frac_data)%*%Q_rot
R2 = get_explained_var(mass_PCs, mass_frac_data)
cat('mass_PCs --> mass_frac_data, R2: ', R2, '\n')
stopifnot(R2>=0.90)
cat('range(mass_PCs): ', range(mass_PCs), '\n')
Q_rot = Q_rot[sort(rownames(Q_rot)),]
write.csv(Q_rot, file=paste0('Q_rot', ifelse(variance_weighted, '_MassR2', ''),'.csv.gz'))

mass_PCs = as.matrix(mass_frac_data)%*%Q_rot %>% as_tibble()
colnames(mass_PCs) = colnames(mass_PCs) %>% paste0('mass_', .)
CPV_sources = as.matrix(souspec_data)%*%Q_rot %>% as_tibble()
colnames(CPV_sources) = colnames(CPV_sources) %>% paste0('source_', .)

Chemtab_data = cbind(Chemtab_data, mass_PCs, CPV_sources)
write.csv(Chemtab_data, file=paste0('Chemtab_data', ifelse(variance_weighted, '_MassR2', ''),'.csv.gz'))
