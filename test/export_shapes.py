from test_utilities import *
import abs.sampling as cs

if __name__ == '__main__':
    curves = {
        "line3d":test_line3d(),
        "line2d":test_line2d(),
        "circle3d":test_circle3d(),
        "circle2d":test_circle2d(),
        "ellipse3d":test_ellipse3d(),
        "ellipse2d":test_ellipse2d(),
        # "bspline_curve3d":test_bspline_curve3d(),
        "bspline_curve2d":test_bspline_curve2d()
    }

    for k in curves:
        c = curves[k]
        t, pts = cs.curve_sampler.uniform_sample(c, 0.01)

        with open(k+".obj", "w") as f:
            if pts.shape[1] == 2:
                [f.write(f"v {pts[i,0]} {pts[i,1]} 0\n") for i in range(pts.shape[0])]
            else:
                [f.write(f"v {pts[i,0]} {pts[i,1]} {pts[i,2]}\n") for i in range(pts.shape[0])]