package dev.langchain4j.store.embedding.milvus;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.*;
import io.milvus.grpc.GetLoadStateResponse;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.util.Collections;
import java.util.List;

import static dev.langchain4j.internal.Utils.randomUUID;
import static java.util.Arrays.asList;
import static org.assertj.core.api.Assertions.assertThat;

@Testcontainers
class MilvusEmbeddingStoreLocalIT extends EmbeddingStoreWithFilteringIT {

    private static final String COLLECTION_NAME = "test_collection";
    private static final String PARTITION_NAME = "test_partition";

    private static final String MILVUS_HOSTS = "127.0.0.1";
    private static final Integer MILVUS_PORT = 19530;
    private static final String MILVUS_DATABASENAME = "test";
    private static final String MILVUS_USERNAME = "admin";
    private static final String MILVUS_PASSWORD = "admin";

    MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
            .host(MILVUS_HOSTS)
            .port(MILVUS_PORT)
            .databaseName(MILVUS_DATABASENAME)
            .username(MILVUS_USERNAME)
            .password(MILVUS_PASSWORD)
            .collectionName(COLLECTION_NAME)
            .dimension(384)
            .retrieveEmbeddingsOnSearch(true)
            .build();

    EmbeddingModel embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();

    @AfterEach
    void afterEach() {
        embeddingStore.dropCollection(COLLECTION_NAME);
    }

    @Override
    protected EmbeddingStore<TextSegment> embeddingStore() {
        return embeddingStore;
    }

    @Override
    protected EmbeddingModel embeddingModel() {
        return embeddingModel;
    }

    @Test
    void should_not_retrieve_embeddings_when_searching() {

        EmbeddingStore<TextSegment> embeddingStore = MilvusEmbeddingStore.builder()
                .host(MILVUS_HOSTS)
                .port(MILVUS_PORT)
                .databaseName(MILVUS_DATABASENAME)
                .username(MILVUS_USERNAME)
                .password(MILVUS_PASSWORD)
                .collectionName(COLLECTION_NAME)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();

        Embedding firstEmbedding = embeddingModel.embed("hello").content();
        Embedding secondEmbedding = embeddingModel.embed("hi").content();
        embeddingStore.addAll(asList(firstEmbedding, secondEmbedding));

        List<EmbeddingMatch<TextSegment>> relevant = embeddingStore.findRelevant(firstEmbedding, 10);
        assertThat(relevant).hasSize(2);
        assertThat(relevant.get(0).embedding()).isNull();
        assertThat(relevant.get(1).embedding()).isNull();
    }

    @Test
    void should_use_partition_searching() {

        MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
                .host(MILVUS_HOSTS)
                .port(MILVUS_PORT)
                .databaseName(MILVUS_DATABASENAME)
                .username(MILVUS_USERNAME)
                .password(MILVUS_PASSWORD)
                .collectionName(COLLECTION_NAME)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();

        Embedding firstEmbedding = embeddingModel.embed("hello").content();
        Embedding secondEmbedding = embeddingModel.embed("hi").content();


        embeddingStore.createPartition(PARTITION_NAME);
        embeddingStore.addAll(asList(firstEmbedding, secondEmbedding), PARTITION_NAME);

        EmbeddingSearchRequest embeddingSearchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(firstEmbedding)
                .maxResults(10)
                .minScore(0.0)
                .build();

        EmbeddingSearchResult<TextSegment> embeddingSearchResult = embeddingStore.search(embeddingSearchRequest, Collections.singletonList(PARTITION_NAME));
        List<EmbeddingMatch<TextSegment>> relevant = embeddingSearchResult.matches();
        assertThat(relevant).hasSize(2);
        assertThat(relevant.get(0).embedding()).isNull();
        assertThat(relevant.get(1).embedding()).isNull();
    }


    @Test
    void should_use_deleteByIds() {
        MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
                .host(MILVUS_HOSTS)
                .port(MILVUS_PORT)
                .databaseName(MILVUS_DATABASENAME)
                .username(MILVUS_USERNAME)
                .password(MILVUS_PASSWORD)
                .collectionName(COLLECTION_NAME)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();
        Embedding firstEmbedding = embeddingModel.embed("hello").content();
        Embedding secondEmbedding = embeddingModel.embed("hi").content();
        embeddingStore.createPartition(PARTITION_NAME);
        List<String> ids = embeddingStore.addAll(asList(firstEmbedding, secondEmbedding), PARTITION_NAME);
        assertThat(ids).isNotNull();

        embeddingStore.deleteByIds(PARTITION_NAME, ids);

    }

    @Test
    void should_use_deletePartition() {

        String partitionName = PARTITION_NAME + randomUUID().replace("-", "");
        MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
                .host(MILVUS_HOSTS)
                .port(MILVUS_PORT)
                .databaseName(MILVUS_DATABASENAME)
                .username(MILVUS_USERNAME)
                .password(MILVUS_PASSWORD)
                .collectionName(COLLECTION_NAME)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();

        Embedding firstEmbedding = embeddingModel.embed("hello").content();
        Embedding secondEmbedding = embeddingModel.embed("hi").content();
        embeddingStore.createPartition(partitionName);
        embeddingStore.addAll(asList(firstEmbedding, secondEmbedding));

        embeddingStore.deletePartition(partitionName);
    }

    @Test
    void should_use_releaseCollection() {

        MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
                .host(MILVUS_HOSTS)
                .port(MILVUS_PORT)
                .databaseName(MILVUS_DATABASENAME)
                .username(MILVUS_USERNAME)
                .password(MILVUS_PASSWORD)
                .collectionName(COLLECTION_NAME)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();
        embeddingStore.releaseCollection();

    }

    @Test
    void should_use_loadCollection() {

        MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
                .host(MILVUS_HOSTS)
                .port(MILVUS_PORT)
                .databaseName(MILVUS_DATABASENAME)
                .username(MILVUS_USERNAME)
                .password(MILVUS_PASSWORD)
                .collectionName(COLLECTION_NAME)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();
        embeddingStore.loadCollection();
    }

    @Test
    void should_use_deleteCollection() {

        MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
                .host(MILVUS_HOSTS)
                .port(MILVUS_PORT)
                .databaseName(MILVUS_DATABASENAME)
                .username(MILVUS_USERNAME)
                .password(MILVUS_PASSWORD)
                .collectionName(COLLECTION_NAME)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();
        embeddingStore.dropCollection();
    }


    @Test
    void should_use_combo() throws InterruptedException {

        MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
                .host(MILVUS_HOSTS)
                .port(MILVUS_PORT)
                .databaseName(MILVUS_DATABASENAME)
                .username(MILVUS_USERNAME)
                .password(MILVUS_PASSWORD)
                .collectionName(COLLECTION_NAME)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();
        GetLoadStateResponse loadState = embeddingStore.getLoadState(null);
        System.out.println(loadState.toString());
        // create partition
        String partitionName_1 = PARTITION_NAME + "001";
        String partitionName_2 = PARTITION_NAME + "002";
        embeddingStore.createPartition(partitionName_1);
        embeddingStore.createPartition(partitionName_2);
        // insert
        Embedding embedding_def = embeddingModel.embed("default_partition name is default").content();
        Embedding embedding_1_1 = embeddingModel.embed("partition_1 name is test_partition_001").content();
        Embedding embedding_1_2 = embeddingModel.embed("partition_001 name is test_partition_001").content();
        Embedding embedding_2_1 = embeddingModel.embed("partition_2 name is test_partition_002").content();
        List<String> embedding_ids_def = embeddingStore.addAll(asList(embedding_def));
        assertThat(embedding_ids_def).isNotNull();
        List<String> embedding_ids_1 = embeddingStore.addAll(asList(embedding_1_1, embedding_1_2), partitionName_1);
        assertThat(embedding_ids_1).isNotNull();
        List<String> embedding_ids_2 = embeddingStore.addAll(asList(embedding_2_1), partitionName_2);
        assertThat(embedding_ids_2).isNotNull();
        // search
        Embedding embedding_search = embeddingModel.embed("partition name").content();
        System.out.println("Initialize complete search");
        search(embedding_search, 4);
        search(embedding_search, 2, partitionName_1);
        search(embedding_search, 1, partitionName_2);
        // delete id
        embeddingStore.deleteByIds(partitionName_1, asList(embedding_ids_1.get(0)));
        embeddingStore.deleteByIds(null, asList(embedding_ids_def.get(0)));
        embeddingStore.deleteByIds(null, asList(embedding_ids_2.get(0)));
        // search
        Thread.sleep(1000); // Note: After the deletion is executed, you must wait for the deletion to complete before querying, otherwise the deleted data will be queried. It can be ignored in actual business.
        System.out.println("Delete By ID complete search");
        search(embedding_search, 1);
        search(embedding_search, 1, partitionName_1);
        search(embedding_search, 1, partitionName_1, partitionName_2);
        // insert and search
        List<String> embedding_ids_2_ = embeddingStore.addAll(asList(embedding_2_1), partitionName_2);
        assertThat(embedding_ids_2_).isNotNull();
        System.out.println("again insert to partition2 search");
        search(embedding_search, 1, partitionName_2);
        search(embedding_search, 2);
        search(embedding_search, 1, partitionName_1);
        // delete partition
        embeddingStore.deletePartition(partitionName_2);
        System.out.println("delete partitionName complete search");
        search(embedding_search, 0, partitionName_2);
        search(embedding_search, 1);
        search(embedding_search, 1, partitionName_1);
        // delete collection collection_c2249186e1c94332b78e08d8a8fc92eb
        embeddingStore.dropCollection(COLLECTION_NAME);
        System.out.println("success");
    }

    private void search(Embedding referenceEmbedding, Integer result_size, String... partitionNames) {
        // Recommendation 2: The partition in partitionNames needs to be verified to exist.
        // Recommendation 1: If the collection has and has only one partition, determine whether there is data in the partition before searching and avoid exceptions as much as possible: Caused by: io.milvus.exception.ServerException: empty expression should be used with limit
        List<EmbeddingMatch<TextSegment>> relevant;
        if (partitionNames != null && partitionNames.length > 0) {
            relevant = embeddingStore.search(EmbeddingSearchRequest.builder().queryEmbedding(referenceEmbedding).maxResults(10).minScore(0.0).build(), asList(partitionNames)).matches();
        } else {
            relevant = embeddingStore.search(EmbeddingSearchRequest.builder().queryEmbedding(referenceEmbedding).maxResults(10).minScore(0.0).build()).matches();
        }
        assertThat(relevant).hasSize(result_size);
        for (int i = 0; i < result_size; i++) {
            System.out.println(relevant.get(i).toString());
            assertThat(relevant.get(i).embedding()).isNotNull();
        }
    }


}