#E-Step for a Document Block
#[a relatively straightforward rewrite of previous
# code with a focus on avoiding unnecessary computation.]

#Input: Documents and Key Global Parameters
#Output: Sufficient Statistics

# Approach:
# First we pre-allocate memory, and precalculate where possible.
# Then for each document we:
#  (1) get document-specific priors, 
#  (2) infer doc parameters, 
#  (3) update global sufficient statistics
# Then the sufficient statistics are returned.

#Let's start by assuming its one beta and we may have arbitrarily subset the number of docs.
estep <- function(documents, beta.index, update.mu, #null allows for intercept only model  
                       beta, lambda.old, mu, sigma, 
                       ncores, verbose) {
  
  # quickly define useful constants
  V <- ncol(beta[[1]])
  K <- nrow(beta[[1]])
  N <- length(documents)
  A <- length(beta)
  ctevery <- ifelse(N>100, floor(N/100), 1)
  if(!update.mu) mu.i <- as.numeric(mu)
  
  # Precalculate common components
  sigobj <- try(chol.default(sigma), silent=TRUE)
  if(class(sigobj)=="try-error") {
    sigmaentropy <- (.5*determinant(sigma, logarithm=TRUE)$modulus[1])
    siginv <- solve(sigma)
  } else {
    sigmaentropy <- sum(log(diag(sigobj)))
    siginv <- chol2inv(sigobj)
  }

  if (ncores == 0) {
    # automatically set number of cores to use
    ncores <- detectCores()
  }

  process <- function(cpuID) {
      start <- ((cpuID - 1) * N) %/% ncores + 1
      end <- (cpuID * N) %/% ncores

      # Initialize Sufficient Statistics 
      sigma.ss <- diag(0, nrow=(K-1))
      beta.ss <- vector(mode="list", length=A)
      for(i in 1:A) {
        beta.ss[[i]] <- matrix(0, nrow=K,ncol=V)
      }
      bound <- vector(length=(end - start + 1))
      lambda <- vector("list", length=(end - start + 1))

      # 3) Document Scheduling
      # For right now we are just doing everything in serial.
      # the challenge with multicore is efficient scheduling while
      # maintaining a small dimension for the sufficient statistics.

      for (i in start:end) {
        #update components
        doc <- documents[[i]]
        words <- doc[1,]
        aspect <- beta.index[i]
        init <- lambda.old[i,]
        if(update.mu) mu.i <- mu[,i]
        beta.i <- beta[[aspect]][,words,drop=FALSE]
        
        #infer the document
        doc.results <- logisticnormalcpp(eta=init, mu=mu.i, siginv=siginv, beta=beta.i, 
                                      doc=doc, sigmaentropy=sigmaentropy)
        
        # update sufficient statistics 
        sigma.ss <- sigma.ss + doc.results$eta$nu
        beta.ss[[aspect]][,words] <- doc.results$phis + beta.ss[[aspect]][,words]
        bound[i - start + 1] <- doc.results$bound
        lambda[[i - start + 1]] <- c(doc.results$eta$lambda)
        if(verbose && i%%ctevery==0) cat(".")
      }

      return(list(sigma=sigma.ss, beta=beta.ss, bound=bound, lambda=lambda))
  }
  t1 <- proc.time()

  batch <- mclapply(1:ncores, process, mc.cores=ncores)
  if(verbose) cat("\n") #add a line break for the next message.

  sigma.ss <- diag(0, nrow=(K-1))
  beta.ss <- vector(mode="list", length=A)
  for(i in 1:A) {
    beta.ss[[i]] <- matrix(0, nrow=K,ncol=V)
  }
  bound <- vector(length=0)
  lambda <- vector("list", length=0)

  for (c in 1:ncores) {
    # update sufficient statistics 
    sigma.ss <- sigma.ss + batch[[c]]$sigma
    for(i in 1:A) {
      beta.ss[[i]] <- beta.ss[[i]] + batch[[c]]$beta[[i]]
    }
    bound <- c(bound, batch[[c]]$bound)
    lambda <- c(lambda, batch[[c]]$lambda)
  }

  # sigma.ss <- batch[[1]]$sigma
  # beta.ss <- batch[[1]]$beta
  # bound <- batch[[1]]$bound
  # lambda <- batch[[1]]$lambda

  #4) Combine and Return Sufficient Statistics
  lambda <- do.call(rbind, lambda)
  return(list(sigma=sigma.ss, beta=beta.ss, bound=bound, lambda=lambda))
}
