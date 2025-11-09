import argparse

from datagen import OriSyntDatasetClass, OptimSyntDatasetClass


def load_args():
    """Function that loads parameters"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imp",
        type=str,
        choices=["ori", "optim"],
        default="ori",
        help="Implementation to profile",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=1,
        help="Number of images to generate for profiling",
    )

    args = parser.parse_args()
    return args


def main():

    # Get parameters
    args = load_args()

    # Create syntclass
    if args.imp == "ori":
        syntclass = OriSyntDatasetClass(
            image_size=512,
            vertex_number=500,
            perlin_noise_coefficient=2.0,
            line_width=0.01,
            start_rad=25,
            line_draw_num=100,
            nami1=5,
            nami2=8,
            oval_rate_x=1.2,
            oval_rate_y=1.8,
        )
    elif args.imp == "optim":
        syntclass = OptimSyntDatasetClass(
            image_size=512,
            vertex_number=500,
            perlin_noise_coefficient=2.0,
            line_width=0.01,
            start_rad=25,
            line_draw_num=100,
            nami1=5,
            nami2=8,
            oval_rate_x=1.2,
            oval_rate_y=1.8,
        )

    # Execute profiling
    times = []
    for i in range(args.N):
        _, t = syntclass.gen_image(idx=i + 8, profile=True)
        times.append(t)
    total = sum(times)
    # Print results
    print(
        f"{args.imp} ({args.N} imgs): {total*1000} ms total {total*1000/args.N} ms per img"
    )


if __name__ == "__main__":
    main()
