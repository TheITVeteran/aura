"""A singleton time axis has the same interpolation as its observed image."""
import pytest

from core.perception.learned_tracking.sampling_a_single_frame import map_coordinates_3d

torch = pytest.importorskip('torch')


@pytest.mark.parametrize('depth', [1, 3])
def test_volume_sampling_matches_the_original_cpu_operation_at_edges_and_outside(depth):
    generator = torch.Generator().manual_seed(72)
    features = torch.rand((2, depth, 9, 13, 4), generator=generator)
    coordinates = torch.tensor([[[0., 0., 0.], [0., 4.5, 8.5], [-10., -4., 20.], [30., 50., -20.]]]).repeat(2, 1, 1)
    image = features.permute(0, 4, 1, 2, 3)
    grid = coordinates.clone()
    grid[..., 0] += .5
    grid = (2. * grid / torch.tensor(image.shape[2:]) - 1.).flip(-1)[:, :, None, None]
    expected = torch.nn.functional.grid_sample(image, grid, mode='bilinear', padding_mode='border',
                                              align_corners=False).squeeze((3, 4)).permute(0, 2, 1)
    actual = map_coordinates_3d(features, coordinates)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(coordinates[0, 0], torch.zeros(3))
