//! VTK writers for visualizing GMRF fields on regular grids.
//!
//! This mirrors the Julia tutorials where posterior samples are exported for
//! external visualization (e.g., ParaView) while keeping a sparse-first workflow.

use crate::types::Vector;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Write scalar fields on a 2D structured grid as legacy VTK STRUCTURED_POINTS (ASCII).
pub fn write_structured_points_2d<W: Write>(
    writer: &mut W,
    title: &str,
    dimensions: (usize, usize),
    origin: (f64, f64),
    spacing: (f64, f64),
    fields: &[(&str, &[f64])],
) -> io::Result<()> {
    let (nx, ny) = dimensions;
    if nx == 0 || ny == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "structured point dimensions must be positive",
        ));
    }
    let npoints = nx * ny;

    writeln!(writer, "# vtk DataFile Version 3.0")?;
    writeln!(writer, "{title}")?;
    writeln!(writer, "ASCII")?;
    writeln!(writer, "DATASET STRUCTURED_POINTS")?;
    writeln!(writer, "DIMENSIONS {} {} 1", nx, ny)?;
    writeln!(writer, "ORIGIN {:.6} {:.6} 0", origin.0, origin.1)?;
    writeln!(writer, "SPACING {:.6} {:.6} 1", spacing.0, spacing.1)?;
    writeln!(writer, "POINT_DATA {}", npoints)?;

    for (name, values) in fields {
        if values.len() != npoints {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "field '{}' has length {}, expected {}",
                    name,
                    values.len(),
                    npoints
                ),
            ));
        }
        writeln!(writer, "SCALARS {} float 1", name)?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for value in *values {
            writeln!(writer, "{:.6}", value)?;
        }
    }

    Ok(())
}

/// Write scalar fields on a 2D structured grid to a legacy VTK STRUCTURED_POINTS file.
pub fn write_structured_points_2d_file(
    path: impl AsRef<Path>,
    title: &str,
    dimensions: (usize, usize),
    origin: (f64, f64),
    spacing: (f64, f64),
    fields: &[(&str, &[f64])],
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_structured_points_2d(&mut writer, title, dimensions, origin, spacing, fields)
}

/// Write scalar fields on a 2D structured grid as a VTU UnstructuredGrid (ASCII XML).
pub fn write_structured_points_2d_vtu<W: Write>(
    writer: &mut W,
    title: &str,
    dimensions: (usize, usize),
    origin: (f64, f64),
    spacing: (f64, f64),
    fields: &[(&str, &[f64])],
) -> io::Result<()> {
    let (nx, ny) = dimensions;
    if nx == 0 || ny == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "structured point dimensions must be positive",
        ));
    }
    let npoints = nx * ny;
    for (name, values) in fields {
        if values.len() != npoints {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "field '{}' has length {}, expected {}",
                    name,
                    values.len(),
                    npoints
                ),
            ));
        }
    }

    let ncells = structured_grid_cell_count(nx, ny);
    writeln!(writer, "<?xml version=\"1.0\"?>")?;
    writeln!(
        writer,
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )?;
    writeln!(writer, "  <!-- {} -->", xml_comment_text(title))?;
    writeln!(writer, "  <UnstructuredGrid>")?;
    writeln!(
        writer,
        "    <Piece NumberOfPoints=\"{npoints}\" NumberOfCells=\"{ncells}\">"
    )?;
    write_vtu_point_data(writer, fields)?;
    writeln!(writer, "      <CellData/>")?;
    write_vtu_points_2d(writer, dimensions, origin, spacing)?;
    write_vtu_cells_2d(writer, dimensions)?;
    writeln!(writer, "    </Piece>")?;
    writeln!(writer, "  </UnstructuredGrid>")?;
    writeln!(writer, "</VTKFile>")?;

    Ok(())
}

/// Write scalar fields on a 2D structured grid to a VTU file.
pub fn write_structured_points_2d_vtu_file(
    path: impl AsRef<Path>,
    title: &str,
    dimensions: (usize, usize),
    origin: (f64, f64),
    spacing: (f64, f64),
    fields: &[(&str, &[f64])],
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_structured_points_2d_vtu(&mut writer, title, dimensions, origin, spacing, fields)
}

/// Write scalar fields on a structured grid as legacy VTK STRUCTURED_POINTS (ASCII).
pub fn write_structured_points<W: Write>(
    writer: &mut W,
    grid_size: usize,
    fields: &[(&str, &Vector)],
) -> io::Result<()> {
    if grid_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "grid_size must be positive",
        ));
    }
    let npoints = grid_size * grid_size;
    let spacing = if grid_size > 1 {
        1.0 / (grid_size as f64 - 1.0)
    } else {
        1.0
    };

    let field_refs = fields
        .iter()
        .map(|(name, values)| (*name, values.as_slice()))
        .collect::<Vec<_>>();
    write_structured_points_2d(
        writer,
        "GMRF structured grid",
        (grid_size, grid_size),
        (0.0, 0.0),
        (spacing, spacing),
        &field_refs,
    )?;

    debug_assert_eq!(npoints, grid_size * grid_size);
    Ok(())
}

/// Write scalar fields on a square structured grid as a VTU UnstructuredGrid.
pub fn write_structured_points_vtu<W: Write>(
    writer: &mut W,
    grid_size: usize,
    fields: &[(&str, &Vector)],
) -> io::Result<()> {
    if grid_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "grid_size must be positive",
        ));
    }
    let spacing = if grid_size > 1 {
        1.0 / (grid_size as f64 - 1.0)
    } else {
        1.0
    };

    let field_refs = fields
        .iter()
        .map(|(name, values)| (*name, values.as_slice()))
        .collect::<Vec<_>>();
    write_structured_points_2d_vtu(
        writer,
        "GMRF structured grid",
        (grid_size, grid_size),
        (0.0, 0.0),
        (spacing, spacing),
        &field_refs,
    )
}

fn structured_grid_cell_count(nx: usize, ny: usize) -> usize {
    if nx > 1 && ny > 1 {
        (nx - 1) * (ny - 1)
    } else if nx > 1 {
        nx - 1
    } else if ny > 1 {
        ny - 1
    } else {
        1
    }
}

fn write_vtu_point_data<W: Write>(writer: &mut W, fields: &[(&str, &[f64])]) -> io::Result<()> {
    if fields.is_empty() {
        writeln!(writer, "      <PointData/>")?;
        return Ok(());
    }

    writeln!(
        writer,
        "      <PointData Scalars=\"{}\">",
        xml_escape(fields[0].0)
    )?;
    for (name, values) in fields {
        writeln!(
            writer,
            "        <DataArray type=\"Float64\" Name=\"{}\" NumberOfComponents=\"1\" format=\"ascii\">",
            xml_escape(name)
        )?;
        write!(writer, "          ")?;
        for value in *values {
            write!(writer, "{value:.12e} ")?;
        }
        writeln!(writer)?;
        writeln!(writer, "        </DataArray>")?;
    }
    writeln!(writer, "      </PointData>")?;
    Ok(())
}

fn write_vtu_points_2d<W: Write>(
    writer: &mut W,
    dimensions: (usize, usize),
    origin: (f64, f64),
    spacing: (f64, f64),
) -> io::Result<()> {
    let (nx, ny) = dimensions;
    writeln!(writer, "      <Points>")?;
    writeln!(
        writer,
        "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">"
    )?;
    for y in 0..ny {
        for x in 0..nx {
            let px = origin.0 + x as f64 * spacing.0;
            let py = origin.1 + y as f64 * spacing.1;
            writeln!(writer, "          {px:.12e} {py:.12e} 0.000000000000e0")?;
        }
    }
    writeln!(writer, "        </DataArray>")?;
    writeln!(writer, "      </Points>")?;
    Ok(())
}

fn write_vtu_cells_2d<W: Write>(writer: &mut W, dimensions: (usize, usize)) -> io::Result<()> {
    let (nx, ny) = dimensions;
    writeln!(writer, "      <Cells>")?;
    writeln!(
        writer,
        "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">"
    )?;
    write_structured_connectivity(writer, nx, ny)?;
    writeln!(writer, "        </DataArray>")?;
    writeln!(
        writer,
        "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">"
    )?;
    write!(writer, "          ")?;
    let width = structured_grid_cell_width(nx, ny);
    for offset in (1..=structured_grid_cell_count(nx, ny)).map(|idx| idx * width) {
        write!(writer, "{offset} ")?;
    }
    writeln!(writer)?;
    writeln!(writer, "        </DataArray>")?;
    writeln!(
        writer,
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">"
    )?;
    write!(writer, "          ")?;
    let cell_type = structured_grid_cell_type(nx, ny);
    for _ in 0..structured_grid_cell_count(nx, ny) {
        write!(writer, "{cell_type} ")?;
    }
    writeln!(writer)?;
    writeln!(writer, "        </DataArray>")?;
    writeln!(writer, "      </Cells>")?;
    Ok(())
}

fn write_structured_connectivity<W: Write>(writer: &mut W, nx: usize, ny: usize) -> io::Result<()> {
    if nx > 1 && ny > 1 {
        for y in 0..ny - 1 {
            for x in 0..nx - 1 {
                let p00 = y * nx + x;
                let p10 = p00 + 1;
                let p11 = (y + 1) * nx + x + 1;
                let p01 = (y + 1) * nx + x;
                writeln!(writer, "          {p00} {p10} {p11} {p01}")?;
            }
        }
    } else if nx > 1 {
        for x in 0..nx - 1 {
            writeln!(writer, "          {x} {}", x + 1)?;
        }
    } else if ny > 1 {
        for y in 0..ny - 1 {
            writeln!(writer, "          {y} {}", y + 1)?;
        }
    } else {
        writeln!(writer, "          0")?;
    }
    Ok(())
}

fn structured_grid_cell_width(nx: usize, ny: usize) -> usize {
    if nx > 1 && ny > 1 {
        4
    } else if nx > 1 || ny > 1 {
        2
    } else {
        1
    }
}

fn structured_grid_cell_type(nx: usize, ny: usize) -> u8 {
    if nx > 1 && ny > 1 {
        9 // VTK_QUAD
    } else if nx > 1 || ny > 1 {
        3 // VTK_LINE
    } else {
        1 // VTK_VERTEX
    }
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn xml_comment_text(value: &str) -> String {
    xml_escape(value).replace("--", "- -")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_structured_points_2d_uses_custom_grid_metadata() {
        let fields = [("field", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..])];
        let mut output = Vec::new();

        write_structured_points_2d(
            &mut output,
            "custom grid",
            (2, 3),
            (10.0, 20.0),
            (0.25, 0.5),
            &fields,
        )
        .expect("write should succeed");

        let content = String::from_utf8(output).expect("VTK should be UTF-8");
        assert!(content.contains("custom grid"));
        assert!(content.contains("DIMENSIONS 2 3 1"));
        assert!(content.contains("ORIGIN 10.000000 20.000000 0"));
        assert!(content.contains("SPACING 0.250000 0.500000 1"));
        assert!(content.contains("POINT_DATA 6"));
    }

    #[test]
    fn write_structured_points_preserves_square_grid_helper() {
        let values = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let fields = [("field", &values)];
        let mut output = Vec::new();

        write_structured_points(&mut output, 2, &fields).expect("write should succeed");

        let content = String::from_utf8(output).expect("VTK should be UTF-8");
        assert!(content.contains("GMRF structured grid"));
        assert!(content.contains("DIMENSIONS 2 2 1"));
        assert!(content.contains("ORIGIN 0.000000 0.000000 0"));
        assert!(content.contains("SPACING 1.000000 1.000000 1"));
    }

    #[test]
    fn write_structured_points_2d_vtu_materializes_unstructured_grid() {
        let fields = [("field", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..])];
        let mut output = Vec::new();

        write_structured_points_2d_vtu(
            &mut output,
            "custom grid",
            (2, 3),
            (10.0, 20.0),
            (0.25, 0.5),
            &fields,
        )
        .expect("write should succeed");

        let content = String::from_utf8(output).expect("VTU should be UTF-8");
        assert!(content.contains("<VTKFile type=\"UnstructuredGrid\""));
        assert!(content.contains("custom grid"));
        assert!(content.contains("NumberOfPoints=\"6\""));
        assert!(content.contains("NumberOfCells=\"2\""));
        assert!(content.contains("<PointData Scalars=\"field\">"));
        assert!(content.contains("Name=\"field\" NumberOfComponents=\"1\""));
        assert!(content.contains("Name=\"connectivity\" format=\"ascii\">\n          0 1 3 2"));
        assert!(content.contains("Name=\"offsets\" format=\"ascii\">\n          4 8 "));
        assert!(content.contains("Name=\"types\" format=\"ascii\">\n          9 9 "));
    }

    #[test]
    fn write_structured_points_vtu_preserves_square_grid_helper() {
        let values = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let fields = [("field", &values)];
        let mut output = Vec::new();

        write_structured_points_vtu(&mut output, 2, &fields).expect("write should succeed");

        let content = String::from_utf8(output).expect("VTU should be UTF-8");
        assert!(content.contains("GMRF structured grid"));
        assert!(content.contains("NumberOfPoints=\"4\""));
        assert!(content.contains("NumberOfCells=\"1\""));
        assert!(content.contains("0.000000000000e0 0.000000000000e0 0.000000000000e0"));
        assert!(content.contains("1.000000000000e0 1.000000000000e0 0.000000000000e0"));
    }

    #[test]
    fn write_structured_points_2d_vtu_rejects_wrong_field_length() {
        let fields = [("field", &[1.0, 2.0, 3.0][..])];
        let mut output = Vec::new();

        let err = write_structured_points_2d_vtu(
            &mut output,
            "custom grid",
            (2, 3),
            (10.0, 20.0),
            (0.25, 0.5),
            &fields,
        )
        .expect_err("field length should be validated");

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }
}
