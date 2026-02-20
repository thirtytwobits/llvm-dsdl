fn main() {
    println!("cargo:rerun-if-changed=transport_shim.c");
    println!("cargo:rerun-if-changed=transport_shim.h");
    println!("cargo:rerun-if-changed=udp_posix.c");
    println!("cargo:rerun-if-changed=udp_posix.h");
    println!("cargo:rerun-if-changed=udpard.c");
    println!("cargo:rerun-if-changed=udpard.h");
    println!("cargo:rerun-if-changed=_udpard_cavl.h");

    cc::Build::new()
        .file("transport_shim.c")
        .file("udp_posix.c")
        .file("udpard.c")
        .include(".")
        .flag_if_supported("-std=c11")
        .compile("cyphal_demo_transport");
}
