package ht.mbds.BarreauSachyEdvalle;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;

import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;
import java.util.Scanner;

public class Test4 {

    public static void main(String[] args) {
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash")
                .logRequestsAndResponses(true)
                .build();

        System.out.println("ChatModel créé");
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStore =
                creerEmbeddingStore("rag.pdf", embeddingModel);

        ContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        System.out.println("ContentRetriever créé");
        QueryRouter queryRouter = query -> {

            String prompt = """
            Est-ce que la requête suivante porte sur l'IA, le RAG ou le fine-tuning ?
            Réponds seulement par oui, non ou peut-être.

            Requête : %s
            """.formatted(query.text());

            String decision = model.chat(prompt);

            System.out.println("Décision du routeur : " + decision);

            if (decision.toLowerCase().contains("oui")
                    || decision.toLowerCase().contains("peut-être")) {
                return List.of(contentRetriever);
            }

            return List.of();
        };

        System.out.println("QueryRouter personnalisé créé");
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        System.out.println("RetrievalAugmentor créé");
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("Assistant créé");
        try (Scanner scanner = new Scanner(System.in)) {

            while (true) {

                System.out.println("==================================================");
                System.out.println("Posez votre question : ");

                String question = scanner.nextLine();

                if (question.isBlank()) {
                    continue;
                }

                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }

                System.out.println("==================================================");

                String reponse = assistant.chat(question);

                System.out.println("Assistant : " + reponse);

                System.out.println("==================================================");
            }
        }
    }
    private static EmbeddingStore<TextSegment> creerEmbeddingStore(
            String nomFichier,
            EmbeddingModel embeddingModel
    ) {
        Document document = ClassPathDocumentLoader.loadDocument(
                nomFichier,
                new ApacheTikaDocumentParser()
        );

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        Response<List<Embedding>> response = embeddingModel.embedAll(segments);

        List<Embedding> embeddings = response.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        return embeddingStore;
    }

}