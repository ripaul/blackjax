
from tensorflow_probability.substrates.jax import distributions as tfd

import jax
import jax.numpy as jnp
import jax.scipy.stats.norm as norm

class sym_dist:
    def __init__(self, dist):
        self._dist = dist
        
    def pdf(self, x):
        return .5 * jnp.maximum(self._dist.pdf(x), self._dist.pdf(-x))
    
    def logpdf(self, x):
        return .5 * jnp.maximum(self._dist.logpdf(x), self._dist.logpdf(-x))
    
    def cdf(self, x):
        return  .5 * (self._dist.cdf(x) - self._dist.cdf(-x) + 1)
    
    def ppf(self, y):
        ppf_pos =  self._dist.ppf(2*y-1)
        ppf_neg = -self._dist.ppf(2*(1-y)-1)
        return jnp.where(y > .5 , ppf_pos, ppf_neg)

#class posnorm:
#    def __init__(self, loc, scale, ):
#        self.loc, self.scale = loc, scale
#        self.F0 = norm.cdf(0.0, loc=loc, scale=scale)
#        
#    def pdf(self, x):
#        base_pdf = norm.pdf(x, loc=self.loc, scale=self.scale)
#        return jnp.where(x > 0, base_pdf / (1.0 - self.F0), 0.0)
#
#    def logpdf(self, x):
#        base_logpdf = norm.logpdf(x, loc=self.loc, scale=self.scale)
#        return jnp.where(x > 0, base_logpdf - jnp.log(1.0 - self.F0), -jnp.inf)
#
#    def cdf(self, x):
#        base_cdf = norm.cdf(x, loc=self.loc, scale=self.scale)
#        cdf_val = (base_cdf - self.F0) / (1.0 - self.F0)
#        return jnp.where(x > 0, cdf_val, 0.0)
#
#    def ppf(self, u):
#        u_adj = u * (1.0 - self.F0) + self.F0
#        return norm.ppf(u_adj, loc=self.loc, scale=self.scale)

class posnorm:
    def __init__(self, loc, scale, ):
        self._dist = tfd.TruncatedNormal(loc=loc, scale=scale, low=0, high=jnp.inf, )
        
    def logpdf(self, x, ):
        return jnp.where(x >= 0., self._dist.log_prob(x, ), -jnp.inf)
    
    def pdf(self, x, ):
        return jnp.where(x >= 0., self._dist.prob(x, ), 0)
    
    def cdf(self, x, ):
        return jnp.where(x >= 0., self._dist.cdf(x, ), 0.)
    
    def ppf(self, x, ):
        return self._dist.quantile(x, )
    
class normchi:
    def __init__(self, df, ):
        _chi = tfd.Chi(df=df)
        #loc, scale = jnp.sqrt(df-1), _chi.stddev()
        loc, scale = _chi.mean(), _chi.stddev()
        self._dist = posnorm(loc=loc, scale=scale, )
        
    def logpdf(self, x, ):
        return jnp.where(x >= 0., self._dist.logpdf(x, ), -jnp.inf)
    
    def pdf(self, x, ):
        return jnp.where(x >= 0., self._dist.pdf(x, ), 0)
    
    def cdf(self, x, ):
        return jnp.where(x >= 0., self._dist.cdf(x, ), 0.)
    
    def ppf(self, x, ):
        return self._dist.ppf(x, )
    
class chi:
    def __init__(self, df, ):
        self._dist = tfd.Chi(df=df, )
        
    def logpdf(self, x, ):
        return jnp.where(x >= 0., self._dist.log_prob(x, ), -jnp.inf)
    
    def pdf(self, x, ):
        return jnp.where(x >= 0., self._dist.prob(x, ), 0)
    
    def cdf(self, x, ):
        return jnp.where(x >= 0., self._dist.cdf(x, ), 0.)
    
    def ppf(self, x, ):
        return self._dist.quantile(x, )
    
    def sample(self, ):
        return self._dist.sample(seed=key)
    
class lognorm:
    def __init__(self, ):
        self._dist = tfd.LogNormal(loc=0., scale=1., )
        
    def logpdf(self, x, ):
        return jnp.where(x >= 0., self._dist.log_prob(x, ), -jnp.inf)
    
    def pdf(self, x, ):
        return jnp.where(x >= 0., self._dist.prob(x, ), 0)
    
    def cdf(self, x, ):
        return jnp.where(x >= 0., self._dist.cdf(x, ), 0.)
    
    def ppf(self, x, ):
        return self._dist.quantile(x, )
    
    def sample(self, key, ):
        return self._dist.sample(seed=key)

