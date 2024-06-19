use tokio::sync::mpsc;
use llm::{Model, ModelParameters};


fn main() {
    const USER_NAME: &str = "###Human";
    const CHARACTER_NAME: &str = "###Asistant";

    let persona = "A chat between a human and an assistant.";
    let history = format!(
        "{CHARACTER_NAME}:Hello - How may I help you today?\n\
        {USER_NAME}:What is the capital of France?\n\
        {CHARACTER_NAME}:Paris is the capital of France.\n"
    );


    let user_message = "Is rust good language?";

    let model = std::path::Path::new("./vicuna-7b-v1.5.ggmlv3.q8_0.bin");

    let llama = llm::load::<llm::models::Llama>(
        
        model,
        
        llm::TokenizerSource::Embedded,

        llm::ModelParameters { prefer_mmap: true, context_size: 5000, lora_adapters: None, use_gpu: true, gpu_layers: Some(10), rope_overrides: None, n_gqa: None },

        llm::load_progress_callback_stdout
    )
    .unwrap_or_else(|err| panic!("Failed to load model! {err}"));

    let mut session = llama.start_session(Default::default());

    session.feed_prompt(
        &llama, 
        format!("{persona}\n{history}").as_str(), 
        &mut Default::default(),
        llm::feed_prompt_callback(|_| {
            Ok::<llm::InferenceFeedback, std::convert::Infallible>(llm::InferenceFeedback::Continue)
        })
    )
    .expect("Failed to ingest initial prompt.");

    let res = session.infer::<std::convert::Infallible>(
    // model to use for text generation
    &llama,
    // randomness provider
    &mut rand::thread_rng(),
    // the prompt to use for text generation, as well as other
    // inference parameters
    &llm::InferenceRequest {
        prompt: llm::Prompt::Text(format!("{USER_NAME}\n{user_message}\n{CHARACTER_NAME}:").as_str()),
        parameters: &Default::default(),
        maximum_token_count: None,
        play_back_previous_tokens: false
    },
    // llm::OutputRequest
    &mut Default::default(),
    // output callback
    inference_callback(String::from(USER_NAME), &mut String::new())
    )
    .unwrap_or_else(|e| panic!("Failed to infer! {e}"));

    println!("{}", res);

}

fn inference_callback<'a>(
    stop_sequence: String,
    buf: &'a mut String,
    //tx: tokio::sync::mpsc::Sender<String>,
    //runtime: &'a mut tokio::runtime::Runtime
) -> impl FnMut(llm::InferenceResponse) -> Result<llm::InferenceFeedback, std::convert::Infallible> + 'a {
    use llm::InferenceFeedback::Halt;
    use llm::InferenceFeedback::Continue;

    move |resp| -> Result<llm::InferenceFeedback, std::convert::Infallible> {match resp {
        llm::InferenceResponse::InferredToken(t) => {
            let mut reverse_buf = buf.clone();
            reverse_buf.push_str(t.as_str());
            if stop_sequence.as_str().eq(reverse_buf.as_str()) {
                buf.clear();
                return Ok(Halt);
            } else if stop_sequence.as_str().starts_with(reverse_buf.as_str()) {
                buf.push_str(t.as_str());
                return Ok(Continue);
            }

            // Clone the string we're going to send
            let text_to_send = if buf.is_empty() {
                t.clone()
            } else {
                reverse_buf
            };

            dbg!("{}", &text_to_send); // I need to find a way to send "text_to_send" from function.
            

            //let tx_cloned = tx.clone();
            //runtime.block_on(async move { // works if main is not async. Implement in separate function ASAP
            //    tx_cloned.send(text_to_send).await.expect("issue sending on channel");
            //});
            
            Ok(Continue)
        }
        llm::InferenceResponse::EotToken => Ok(Halt),
        _ => Ok(Continue),
    }}
}