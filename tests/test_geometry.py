"""Tests for geometry utilities."""

import numpy as np
import pytest

from video_pii.utils.geometry import expand_bbox, iou, smooth_bbox


class TestIoU:
    """Tests for IoU calculation."""
    
    def test_identical_boxes(self):
        box = (10, 10, 50, 50)
        assert iou(box, box) == pytest.approx(1.0)
    
    def test_no_overlap(self):
        box_a = (0, 0, 10, 10)
        box_b = (20, 20, 30, 30)
        assert iou(box_a, box_b) == pytest.approx(0.0)
    
    def test_partial_overlap(self):
        # Two 10x10 boxes with 5x10 overlap
        box_a = (0, 0, 10, 10)   # area = 100
        box_b = (5, 0, 15, 10)   # area = 100
        # intersection = 5x10 = 50
        # union = 100 + 100 - 50 = 150
        # iou = 50/150 = 1/3
        assert iou(box_a, box_b) == pytest.approx(1/3)
    
    def test_one_inside_other(self):
        box_a = (0, 0, 100, 100)  # area = 10000
        box_b = (25, 25, 75, 75)  # area = 2500, fully inside
        # intersection = 2500
        # union = 10000
        # iou = 2500/10000 = 0.25
        assert iou(box_a, box_b) == pytest.approx(0.25)
    
    def test_touching_edge(self):
        box_a = (0, 0, 10, 10)
        box_b = (10, 0, 20, 10)  # Touching at x=10
        assert iou(box_a, box_b) == pytest.approx(0.0)
    
    def test_zero_area_box(self):
        box_a = (10, 10, 10, 10)  # Zero area
        box_b = (0, 0, 20, 20)
        assert iou(box_a, box_b) == pytest.approx(0.0)
    
    def test_numpy_arrays(self):
        box_a = np.array([0, 0, 10, 10])
        box_b = np.array([5, 0, 15, 10])
        assert iou(box_a, box_b) == pytest.approx(1/3)


class TestSmoothBbox:
    """Tests for bbox smoothing."""
    
    def test_alpha_one_returns_new_box(self):
        prev = (0, 0, 10, 10)
        new = (20, 20, 30, 30)
        result = smooth_bbox(prev, new, alpha=1.0)
        assert result == (20.0, 20.0, 30.0, 30.0)
    
    def test_alpha_zero_returns_prev_box(self):
        prev = (0, 0, 10, 10)
        new = (20, 20, 30, 30)
        result = smooth_bbox(prev, new, alpha=0.0)
        assert result == (0.0, 0.0, 10.0, 10.0)
    
    def test_alpha_half_returns_average(self):
        prev = (0, 0, 10, 10)
        new = (10, 10, 20, 20)
        result = smooth_bbox(prev, new, alpha=0.5)
        assert result == (5.0, 5.0, 15.0, 15.0)
    
    def test_with_numpy_arrays(self):
        prev = np.array([0, 0, 10, 10])
        new = np.array([10, 10, 20, 20])
        result = smooth_bbox(prev, new, alpha=0.5)
        assert result == (5.0, 5.0, 15.0, 15.0)
    
    def test_returns_tuple(self):
        result = smooth_bbox((0, 0, 10, 10), (10, 10, 20, 20), 0.5)
        assert isinstance(result, tuple)
        assert len(result) == 4


class TestExpandBbox:
    """Tests for bbox expansion."""
    
    def test_no_padding(self):
        bbox = (10, 10, 50, 50)
        result = expand_bbox(bbox, padding=0.0, img_width=100, img_height=100)
        assert result == (10, 10, 50, 50)
    
    def test_with_padding(self):
        bbox = (20, 20, 60, 60)  # 40x40 box
        # 30% padding = 12px on each side
        result = expand_bbox(bbox, padding=0.3, img_width=100, img_height=100)
        assert result == (8, 8, 72, 72)
    
    def test_clips_to_image_bounds(self):
        bbox = (0, 0, 20, 20)  # At corner
        result = expand_bbox(bbox, padding=0.5, img_width=100, img_height=100)
        # Should clip negative coordinates to 0
        assert result[0] >= 0
        assert result[1] >= 0
    
    def test_clips_to_max_bounds(self):
        bbox = (80, 80, 100, 100)  # At corner
        result = expand_bbox(bbox, padding=0.5, img_width=100, img_height=100)
        assert result[2] <= 100
        assert result[3] <= 100

