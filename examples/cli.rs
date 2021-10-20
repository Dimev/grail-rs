use std::env;

fn main() {
    // get the command line args
    let args: Vec<String> = env::args().collect();

    // display them
    println!("Args: {:?}", args);
}
