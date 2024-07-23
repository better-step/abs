import numpy as np
import math as m


def blue_noise_downsample(pts, radius):
    assert(pts.shape[1] == 3) # "Only 3D embeddings allowed"

    # minimum radius
    min_r = radius

    # cell size based on 3D distance
    # It works reasonably well (but is probably biased to use s=2*r/√3 here and
    # g=1 in the outer loop below.
    #
    # One thing to try would be to store a list in S (rather than a single point)
    # or equivalently a mask over M and just use M as a generic spatial hash
    # (with arbitrary size) and then tune its size (being careful to make g a
    # function of r and s; and removing the `if S=-1 checks`)
    s = radius/m.sqrt(3.0)

    # Make a uniform random sampling with 30*expected_number_of_points.
    nx = pts.shape[0]

    # Rescale so that s = 1
    Xs = ((pts-np.min(pts, axis=0))/s).astype(int)
    w = np.max(Xs) + 1
    SortIdx = np.argsort(Xs[:, 0])  # igl::sortrows(decltype(Xs)(Xs), true, Xs, SortIdx)
    Xsorted = Xs[SortIdx, :]


    # Initialization
    M = {}
    S = {}
    # attempted to seed
    A = {}

    for i in range(Xs.shape[0]):
        k = _blue_noise_key(w, Xs[i,0], Xs[i,1],Xs[i,2])

        if k in M:
            M[k].append(i)
        else:
            M[k] = [i]

        S[k] = -1
        A[k] = False


    active = []
    # precompute r²
    # Q: is this necessary?
    rr = radius**2
    collected = []

    # Just take the first point as the initial seed
    def initialize():
        for k in M:
            # Haven't placed in this cell yet
            if S[k]<0:
                if _activate(Xsorted,Xs,rr,-1,w,k,M,S,active):
                    return True

        return False

    # important if mesh contains many connected components
    while initialize():
        while len(active)>0:
            _step(Xsorted,Xs,rr,w, M, S,active,collected);


    n = len(collected)
    XI = np.zeros((n,1))
    for i in range(n):
        c = collected[i]
        XI[i] = SortIdx[c]

    return XI

def downsample(pts, target_num_samples, sample_num_tolerance=0.04):
    """Downsample a point cloud using Poisson disk sampling."""

    if target_num_samples <= 0:
        raise ValueError("target_num_samples must be positive")

    if sample_num_tolerance > 1.0 or sample_num_tolerance <= 0.0:
        raise ValueError("sample_num_tolerance must be in (0,1]")

    if target_num_samples >= pts.shape[0]:
        return np.arange(pts.shape[0])


    num_samples_min = int(target_num_samples * (1.0 - sample_num_tolerance))
    num_samples_max = int(target_num_samples * (1.0 + sample_num_tolerance));

    bmin = np.min(pts, axis=0)
    bmax = np.max(pts, axis=0)
    bbsize = np.linalg.norm(bmax - bmin)

    range_min_rad = bbsize / 50.0
    range_max_rad = bbsize / 50.0

    range_min_rad_num = -1
    range_max_rad_num = -1

    while True:
        # ret_i.conservativeResize(0, 1);
        range_min_rad /= 2.0

        ret_i = blue_noise_downsample(pts, range_min_rad)
        range_min_rad_num = ret_i.shape[0]

        if range_min_rad_num >= target_num_samples:
            break

    while True:
        # ret_i.conservativeResize(0, 1);
        range_max_rad *= 2.0;
        ret_i = blue_noise_downsample(pts, range_max_rad)
        range_max_rad_num = ret_i.shape[0]

        if range_max_rad_num <= target_num_samples:
            break

    current_rad = range_max_rad
    iter_count = 0

    while iter_count < 20 and (ret_i.shape[0] < num_samples_min or ret_i.shape[0] > num_samples_max):
        iter_count += 1
        current_rad = (range_min_rad + range_max_rad) / 2.0

        ret_i = blue_noise_downsample(pts, current_rad)

        if ret_i.shape[0] > target_num_samples:
            range_min_rad = current_rad
            range_min_rad_num = ret_i.shape[0]

        if ret_i.rows() < target_num_samples:
            range_max_rad = current_rad
            range_max_rad_num = ret_i.shape[0]


    return ret_i


def _blue_noise_key(w, x, y, z):
    return x+w*(y+w*z)

def _blue_noise_far_enough(X, Xs, S, rr, w, i):
    xi = Xs[i,0]
    yi = Xs[i,1]
    zi = Xs[i,2]

    g = 2 # ceil(r/s)
    for x in range(max(xi-g,0), min(xi+g,w-1)+1):
        for y in range(max(yi-g,0),min(yi+g,w-1)+1):
            for z in range(max(zi-g,0),min(zi+g,w-1)+1):
                if x!=xi or y!=yi or z!=zi:
                    nk = _blue_noise_key(w,x,y,z);
                    # have already selected from this cell
                    if nk in S and S[nk] >= 0:
                        ni = S[nk]
                        # too close
                        if np.linalg.norm(X[i,:]-X[ni,:])**2 < rr:
                            return False

    return True

def _activate(X, Xs, rr, i, w, nk, M, S, active):
    Mvec = M[nk]
    miter = 0

    while miter < len(Mvec):
      mi = Mvec[miter]
      # mi is our candidate sample. Is it far enough from all existing samples?
      if i>=0 and np.linalg.norm(X[i,:]-X[mi,:])**2 > 4.*rr:
        # too far skip (reject)
        miter += 1
      elif(_blue_noise_far_enough(X,Xs,S,rr,w,mi)):
        active.append(mi)
        S[nk] = mi
        return True
      else:
        # remove forever (instead of incrementing we swap and eat from the back)

        miter = Mvec[-1]
        was_last = miter+1 == len(Mvec)
        Mvec.pop()
        if was_last:
          # popping from the vector can invalidate the iterator, if it was
          # pointing to the last element that was popped. Alternatively,
          # one could use indices directly...
          miter = len(Mvec)

    return False


def _step(X, Xs, rr, w, M, S, active, collected):
    if len(active):
        return False

    # random entry
    e = np.random.uniform(0, active.size()-1)
    i = active[e]

    xi = Xs[i,0]
    yi = Xs[i,1]
    zi = Xs[i,2]

    # cell indices of neighbors
    g = 4

    N = []

    for x in range(max(xi-g,0),min(xi+g,w-1)+1):
        for y in range(max(yi-g,0),min(yi+g,w-1)+1):
            for z in range(max(zi-g,0),min(zi+g,w-1)+1):
                if x!=xi or y!=yi or z!=zi:
                    nk = _blue_noise_key(w,x,y,z);

                    # haven't yet selected from this cell?
                    if nk in S and S[nk] < 0:
                        assert(nk in M)
                        N.append(nk)

    # randomize order: this might be a little paranoid...
    np.random.shuffle(N)
    found = False
    for nk in N:
      assert(nk in M)
      if _activate(X,Xs,rr,i,w,nk,M,S,active):
        found = True
        break

    if not found:
      # remove i from active list
      # https://stackoverflow.com/a/60765833/148668
      collected.append(i)
      active[e] = active[-1]
      active.pop()

    return True
