recursive.centr <- function(X, n) {
    if (dim(X)[1] == 2) {
        return(X[1, 2] + 1)
    }
    for (i in 1:dim(X)[1]) {
        if (i != n && X[n, i] != 0) {
            Mt <- X
            Mt[n, ] <- Mt[n, ] + Mt[i, ]
            Mt[, n] <- Mt[, n] + Mt[, i]
            v <- Mt[n, n] / 2
            Mt[n, n] <- 0
            Mt <- Mt[-i, ]
            Mt <- Mt[, -i]
            X[n, i] <- 0
            X[i, n] <- 0
            return(recursive.centr(X, n) + v * recursive.centr(Mt, n - (i < n)))
        }
    }
    return(1)
}

centr_Trees <- function(X) {
    r <- 1:dim(X)[1]
    for (i in 1:dim(X)[1]) {
        r[i] <- log2(recursive.centr(X, i))
    }
    return(r)
}

g2 <- graph(edges = c(1, 2, 2, 3, 1, 4, 4, 3, 1, 5, 5, 3, 1, 6, 6, 3, 3, 7, 7, 8, 8, 9, 3, 8), n = 9, directed = F)
c2 <- centr(as_adj(g2))
