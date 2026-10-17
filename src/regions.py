from math import isfinite


def scale_normalized_regions(normalized_regions, frame_width, frame_height):
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame width and height must both be positive.")

    regions = {}
    for region_name, normalized_coordinates in normalized_regions.items():
        if len(normalized_coordinates) != 4:
            raise ValueError(f"Region '{region_name}' must have four coordinates.")

        left_ratio, top_ratio, right_ratio, bottom_ratio = normalized_coordinates
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            for value in normalized_coordinates
        ):
            raise ValueError(
                f"Region '{region_name}' coordinates must be finite numbers."
            )
        if not 0 <= left_ratio < right_ratio <= 1:
            raise ValueError(f"Region '{region_name}' has invalid horizontal coordinates.")
        if not 0 <= top_ratio < bottom_ratio <= 1:
            raise ValueError(f"Region '{region_name}' has invalid vertical coordinates.")

        regions[region_name] = (
            round(left_ratio * frame_width),
            round(top_ratio * frame_height),
            round(right_ratio * frame_width),
            round(bottom_ratio * frame_height),
        )

    return regions


def split_regions_into_tiles(regions, tile_rows, tile_columns, tile_overlap_ratio):
    if tile_rows < 1 or tile_columns < 1:
        raise ValueError("Tile rows and columns must both be at least one.")
    if not 0 <= tile_overlap_ratio < 0.5:
        raise ValueError("Tile overlap ratio must be at least zero and less than 0.5.")

    tiled_regions = {}
    for region_name, (left, top, right, bottom) in regions.items():
        region_width = right - left
        region_height = bottom - top
        tile_width = region_width / tile_columns
        tile_height = region_height / tile_rows
        horizontal_overlap = round(tile_width * tile_overlap_ratio)
        vertical_overlap = round(tile_height * tile_overlap_ratio)

        for row_index in range(tile_rows):
            for column_index in range(tile_columns):
                tile_left = round(left + column_index * tile_width)
                tile_top = round(top + row_index * tile_height)
                tile_right = round(left + (column_index + 1) * tile_width)
                tile_bottom = round(top + (row_index + 1) * tile_height)

                if column_index > 0:
                    tile_left -= horizontal_overlap
                if column_index < tile_columns - 1:
                    tile_right += horizontal_overlap
                if row_index > 0:
                    tile_top -= vertical_overlap
                if row_index < tile_rows - 1:
                    tile_bottom += vertical_overlap

                tiled_regions[f"{region_name}_{row_index}_{column_index}"] = (
                    tile_left,
                    tile_top,
                    tile_right,
                    tile_bottom,
                )

    return tiled_regions
