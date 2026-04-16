from core import FaceTracker


class TestIoU:
    def test_no_overlap(self):
        assert FaceTracker._iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_full_overlap(self):
        assert abs(FaceTracker._iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9

    def test_partial_overlap(self):
        iou = FaceTracker._iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert 0.0 < iou < 1.0
        # intersection = 5*5=25, union = 100+100-25=175
        assert abs(iou - 25.0 / 175.0) < 1e-6

    def test_degenerate_box(self):
        assert FaceTracker._iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0


class TestUpdate:
    def test_first_frame_assigns_sequential_ids(self):
        ft = FaceTracker()
        ids = ft.update([[10, 10, 100, 100], [200, 200, 300, 300]])
        assert ids == [0, 1]

    def test_same_bbox_keeps_id(self):
        ft = FaceTracker()
        ft.update([[100, 100, 200, 200]])
        ids = ft.update([[102, 98, 198, 203]])  # slight shift
        assert ids == [0]

    def test_swapped_bboxes_keep_original_ids(self):
        ft = FaceTracker()
        ft.update([[10, 10, 100, 100], [300, 300, 400, 400]])
        # Swap the two faces in detection order
        ids = ft.update([[300, 300, 400, 400], [10, 10, 100, 100]])
        assert ids == [1, 0]  # IDs follow the face, not the index

    def test_new_face_gets_new_id(self):
        ft = FaceTracker()
        ft.update([[10, 10, 100, 100]])
        ids = ft.update([[10, 10, 100, 100], [500, 500, 600, 600]])
        assert ids[0] == 0  # existing
        assert ids[1] == 1  # new

    def test_stale_track_pruned(self):
        ft = FaceTracker(max_missing_frames=2)
        ft.update([[100, 100, 200, 200]])
        ft.update([])  # missing 1
        ft.update([])  # missing 2
        ft.update([])  # missing 3 → pruned
        assert ft.pruned_ids() == [0]
        assert 0 not in ft.tracks

    def test_track_survives_brief_absence(self):
        ft = FaceTracker(max_missing_frames=5)
        ft.update([[100, 100, 200, 200]])
        ft.update([])  # missing 1
        ft.update([])  # missing 2
        ids = ft.update([[100, 100, 200, 200]])  # returns
        assert ids == [0]  # same ID
