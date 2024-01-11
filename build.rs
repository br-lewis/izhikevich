use std::env;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::Path;

const SHADER_FILE: &str = "izhikevich.comp";
const COMPILED_NAME: &str = "izhikevich.comp.spv";

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join(COMPILED_NAME);

    let shader_path = Path::new("shaders").join(SHADER_FILE);
    let shader_data = compile_shader(&shader_path);

    let mut f = File::create(&dest_path).expect("unable to create shader target file");
    f.write_all(shader_data.as_binary_u8())
        .expect("unable to write shader data to file");
}

fn compile_shader(shader: &Path) -> shaderc::CompilationArtifact {
    let mut f = File::open(shader).expect("unable to open shader file");
    let mut buf: String = String::new();
    f.read_to_string(&mut buf)
        .expect("unable to read shader file");

    let compiler = shaderc::Compiler::new().expect("error creating shader compiler");
    let options = shaderc::CompileOptions::new().expect("error creating shader compiler options");

    match compiler.compile_into_spirv(
        &buf,
        shaderc::ShaderKind::Compute,
        SHADER_FILE,
        "main",
        Some(&options),
    ) {
        Ok(data) => data,
        Err(err) => {
            panic!("error compiling shader:\n{}", err);
        }
    }
}
