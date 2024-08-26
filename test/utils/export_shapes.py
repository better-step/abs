from utils.test_utilities import *
import abs.sampling as cs

if __name__ == '__main__':
    curves = {
        "line3d": line3d(),
        "line2d": line2d(),
        "circle3d": circle3d(),
        "circle2d": circle2d(),
        "ellipse3d": ellipse3d(),
        "ellipse2d": ellipse2d(),
        # "bspline_curve3d": bspline_curve3d(),
        "bspline_curve2d": bspline_curve2d()
    }

    for k in curves:
        c = curves[k]
        t, pts = cs.curve_sampler.uniform_sample(c, 0.01)

        with open(k+".obj", "w") as f:
            if pts.shape[1] == 2:
                [f.write(f"v {pts[i,0]} {pts[i,1]} 0\n") for i in range(pts.shape[0])]
            else:
                [f.write(f"v {pts[i,0]} {pts[i,1]} {pts[i,2]}\n") for i in range(pts.shape[0])]